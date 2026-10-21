"""Evaluate TactX alignment, retrieval, probes, reconstruction, and optional dynamic data.

Run ``python -m scripts.evaluate --help`` for checkpoint and data overrides.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    r2_score, mean_squared_error, mean_absolute_error,
)
import inspect
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from joint_embedding_learning.data.datasets import (
    get_cross_modal_tri_datasets,
)
from joint_embedding_learning.utils.models_setup import get_model

_SENSORS = ('D', 'E', 'F')
_PAIRS = ('de', 'ef', 'fd')
_PAIR_SENSORS = {'de': ('D', 'E'), 'ef': ('E', 'F'), 'fd': ('F', 'D')}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _shape_match_recon(recon, target):
    """Reconcile singleton channel mismatch (FlexiTac decoder vs dataset)."""
    if recon.shape == target.shape:
        return recon
    if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
        return recon.squeeze(1)
    if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
        return recon.unsqueeze(1)
    return recon


@torch.no_grad()
def extract_pair_mu(dataset, pair_name, model, device, batch_size=128,
                    return_meta=False):
    """Extract paired μ. If return_meta=True, also returns per-sample
    (object_id, episode_id, frame_id) tensors from the dataset labels —
    used by the contact-classification probe to align μ with GT labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    mu_a_all, mu_b_all = [], []
    obj_all, ep_all, fr_all = [], [], []
    for (xa, xb), labels in tqdm(loader, desc=f"paired_{pair_name}", leave=False):
        out = model.forward_pair(xa.to(device), xb.to(device), pair_name)
        mu_a_all.append(out['mu_a'].cpu())
        mu_b_all.append(out['mu_b'].cpu())
        if return_meta:
            obj_id, ep_id, fr_id = labels
            obj_all.append(obj_id.cpu())
            ep_all.append(ep_id.cpu())
            fr_all.append(fr_id.cpu())
    mu_a = torch.cat(mu_a_all)
    mu_b = torch.cat(mu_b_all)
    if return_meta:
        return (mu_a, mu_b,
                torch.cat(obj_all), torch.cat(ep_all), torch.cat(fr_all))
    return mu_a, mu_b


@torch.no_grad()
def extract_pair_recon(dataset, pair_name, model, device, batch_size=128):
    """Extract L1 recon errors per direction (self_a, self_b, cross_ab, cross_ba)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    errs = {'self_a': [], 'self_b': [], 'cross_ab': [], 'cross_ba': []}
    for (xa, xb), _ in tqdm(loader, desc=f"recon_{pair_name}", leave=False):
        out = model.forward_pair(xa.to(device), xb.to(device), pair_name)
        xa_d, xb_d = xa.to(device), xb.to(device)
        errs['self_a'].append(F.l1_loss(_shape_match_recon(out['recon_aa'], xa_d), xa_d, reduction='none').mean(dim=list(range(1, xa_d.dim()))).cpu())
        errs['self_b'].append(F.l1_loss(_shape_match_recon(out['recon_bb'], xb_d), xb_d, reduction='none').mean(dim=list(range(1, xb_d.dim()))).cpu())
        errs['cross_ab'].append(F.l1_loss(_shape_match_recon(out['recon_ab'], xb_d), xb_d, reduction='none').mean(dim=list(range(1, xb_d.dim()))).cpu())
        errs['cross_ba'].append(F.l1_loss(_shape_match_recon(out['recon_ba'], xa_d), xa_d, reduction='none').mean(dim=list(range(1, xa_d.dim()))).cpu())
    return {k: torch.cat(v).mean().item() for k, v in errs.items()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine_pos_neg(mu_a, mu_b, sample_neg=5000):
    a = F.normalize(mu_a, dim=1)
    b = F.normalize(mu_b, dim=1)
    pos = (a * b).sum(dim=1).cpu().numpy()
    N = a.shape[0]
    idx_a = np.random.randint(0, N, sample_neg)
    idx_b = np.random.randint(0, N, sample_neg)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N
    neg = (a[idx_a] * b[idx_b]).sum(dim=1).cpu().numpy()
    return pos, neg


def transitive_df_cosine(mu_D_de, mu_E_de, mu_E_ef, mu_F_ef, sample_neg=5000):
    e_de_n = F.normalize(mu_E_de, dim=1)
    e_ef_n = F.normalize(mu_E_ef, dim=1)
    nn_idx = (e_ef_n @ e_de_n.t()).argmax(dim=1)
    mu_D_match = mu_D_de[nn_idx]
    d_n = F.normalize(mu_D_match, dim=1)
    f_n = F.normalize(mu_F_ef, dim=1)
    pos = (d_n * f_n).sum(dim=1).cpu().numpy()
    N = d_n.shape[0]
    idx_a = np.random.randint(0, N, sample_neg)
    idx_b = np.random.randint(0, N, sample_neg)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N
    neg = (d_n[idx_a] * f_n[idx_b]).sum(dim=1).cpu().numpy()
    return pos, neg


def mmd_rbf_multi(x, y, bandwidths=(1.0, 2.0, 4.0, 8.0, 16.0)):
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)
    mmd2 = 0.0
    for s in bandwidths:
        mmd2 += (torch.exp(-xx / (2 * s)).mean()
                 + torch.exp(-yy / (2 * s)).mean()
                 - 2 * torch.exp(-xy / (2 * s)).mean())
    return float(mmd2 / len(bandwidths))


def latent_l2_positive(mu_a, mu_b):
    """Per-sample L2 distance between positive pairs in latent z (μ) space."""
    d = (mu_a - mu_b).pow(2).sum(dim=1).sqrt()
    return d.mean().item(), d.std().item(), d.shape[0]


def _build_sensor_dataset(mu_pairs):
    """Flatten {pk: (μ_a, μ_b)} into (X, y) with sensor labels D/E/F."""
    feats, labels = [], []
    for pk in _PAIRS:
        ma, mb = mu_pairs[pk]
        sa, sb = _PAIR_SENSORS[pk]
        feats.append(ma)
        labels.extend([sa] * len(ma))
        feats.append(mb)
        labels.extend([sb] * len(mb))
    X = torch.cat(feats, dim=0).numpy()
    y = np.array(labels)
    return X, y


def sensor_probe(mu_pairs_tr, mu_pairs_v, normalize=False, seed=0):
    """LogReg probe: predict sensor (D/E/F) from frozen μ. Train on tr, eval on val."""
    X_tr, y_tr = _build_sensor_dataset(mu_pairs_tr)
    X_v, y_v = _build_sensor_dataset(mu_pairs_v)
    if normalize:
        X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-12)
        X_v = X_v / (np.linalg.norm(X_v, axis=1, keepdims=True) + 1e-12)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_v)
    acc = accuracy_score(y_v, y_pred)
    bal_acc = balanced_accuracy_score(y_v, y_pred)
    cm = confusion_matrix(y_v, y_pred, labels=list(_SENSORS))
    return acc, bal_acc, cm, (len(y_tr), len(y_v))


# ---------------------------------------------------------------------------
# Contact classification probe (uses labels from build_contact_labels.py)
# ---------------------------------------------------------------------------

_PAIR_DIR = {'de': 'de_pair', 'ef': 'ef_pair', 'fd': 'fd_pair'}


def _unwrap_dataset(d):
    """Peel through torch.utils.data.Subset wrappers to reach the base
    CrossModalPairDataset (which holds .object_names)."""
    while isinstance(d, Subset):
        d = d.dataset
    return d


def _load_pair_contact_labels(label_root, pair_name, object_names):
    """Build {(obj_id, ep_id, frame_id): contact_pair_int} for one pair.

    `object_names` is the index→name list from the underlying dataset
    (order matters: obj_id == index into this list)."""
    pair_dir = _PAIR_DIR[pair_name]
    label_map = {}
    for obj_id, obj in enumerate(object_names):
        ep_dir = os.path.join(label_root, pair_dir, obj)
        if not os.path.isdir(ep_dir):
            continue
        for f in sorted(glob.glob(os.path.join(ep_dir, 'ep*.pt'))):
            d = torch.load(f, map_location='cpu', weights_only=False)
            ep = int(d['episode'])
            frames = np.asarray(d['frames'])
            cp = np.asarray(d['contact_pair']).astype(np.int64)
            for fr, c in zip(frames, cp):
                label_map[(obj_id, ep, int(fr))] = int(c)
    return label_map


def _gather_contact_y(meta, label_map, default=-1):
    """Map sample-level (obj_id, ep_id, fr_id) tensors -> int8 label array."""
    obj_t, ep_t, fr_t = meta
    n = obj_t.shape[0]
    y = np.full(n, default, dtype=np.int64)
    obj_np = obj_t.numpy()
    ep_np = ep_t.numpy()
    fr_np = fr_t.numpy()
    for i in range(n):
        key = (int(obj_np[i]), int(ep_np[i]), int(fr_np[i]))
        if key in label_map:
            y[i] = label_map[key]
    return y


def _load_pair_score_labels(label_root, pair_name, object_names,
                            score_key='norm'):
    """Like _load_pair_contact_labels but returns continuous scores per sensor.

    score_key='norm'   -> per-episode min-max normalized score in [0,1]
    score_key='scores' -> raw mean-abs-diff score (unbounded, episode-dependent)

    Returns {(obj_id, ep_id, frame_id): (score_a, score_b)}.
    """
    pair_dir = _PAIR_DIR[pair_name]
    ka, kb = f'{score_key}_a', f'{score_key}_b'
    label_map = {}
    for obj_id, obj in enumerate(object_names):
        ep_dir = os.path.join(label_root, pair_dir, obj)
        if not os.path.isdir(ep_dir):
            continue
        for f in sorted(glob.glob(os.path.join(ep_dir, 'ep*.pt'))):
            d = torch.load(f, map_location='cpu', weights_only=False)
            ep = int(d['episode'])
            frames = np.asarray(d['frames'])
            sa = np.asarray(d[ka]).astype(np.float32)
            sb = np.asarray(d[kb]).astype(np.float32)
            for fr, va, vb in zip(frames, sa, sb):
                label_map[(obj_id, ep, int(fr))] = (float(va), float(vb))
    return label_map


def _build_persensor_score_data(mu_pairs, meta, object_names_per_pair,
                                label_root, score_key='norm'):
    """Aggregate {sensor: (X, y)} where each frame contributes ONLY to its own
    sensor (no pair sharing). Same frame may appear in 2 different sensor
    arrays via mu_a / mu_b of overlapping pairs (e.g. D shows up in both de.a
    and fd.b) — that's correct, those are different μ measurements of the
    same physical event."""
    bucket = {s: [[], []] for s in _SENSORS}    # sensor -> [Xs_list, ys_list]
    skipped = 0
    for pk in _PAIRS:
        labels = _load_pair_score_labels(
            label_root, pk, object_names_per_pair[pk], score_key=score_key,
        )
        if not labels:
            continue
        ma, mb = mu_pairs[pk]
        obj_t, ep_t, fr_t = meta[pk]
        obj_np, ep_np, fr_np = obj_t.numpy(), ep_t.numpy(), fr_t.numpy()
        sa_name, sb_name = _PAIR_SENSORS[pk]

        n = ma.shape[0]
        va = np.zeros(n, dtype=np.float32)
        vb = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            key = (int(obj_np[i]), int(ep_np[i]), int(fr_np[i]))
            if key in labels:
                va[i], vb[i] = labels[key]
                valid[i] = True
            else:
                skipped += 1

        if not valid.any():
            continue
        bucket[sa_name][0].append(ma[valid].numpy())
        bucket[sa_name][1].append(va[valid])
        bucket[sb_name][0].append(mb[valid].numpy())
        bucket[sb_name][1].append(vb[valid])

    out = {}
    for s in _SENSORS:
        if bucket[s][0]:
            out[s] = (np.concatenate(bucket[s][0], axis=0),
                      np.concatenate(bucket[s][1], axis=0))
    return out, skipped


def contact_score_probe(mu_pairs_tr, mu_pairs_v, meta_tr, meta_v,
                        object_names_per_pair, label_root,
                        normalize=False, seed=0, alpha=1.0,
                        score_key='norm'):
    """Per-sensor Ridge regression: predict (normalized) contact score from μ.

    No pair sharing — each sensor's μ vectors are paired only with its own
    score. Returns {sensor: {'r2', 'mse', 'mae', 'n_train', 'n_val'}}.
    """
    data_tr, _ = _build_persensor_score_data(
        mu_pairs_tr, meta_tr, object_names_per_pair, label_root,
        score_key=score_key,
    )
    data_v, _ = _build_persensor_score_data(
        mu_pairs_v, meta_v, object_names_per_pair, label_root,
        score_key=score_key,
    )
    if not data_tr or not data_v:
        return {}

    results = {}
    for s in _SENSORS:
        if s not in data_tr or s not in data_v:
            continue
        X_tr, y_tr = data_tr[s]
        X_v, y_v = data_v[s]
        if normalize:
            X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-12)
            X_v = X_v / (np.linalg.norm(X_v, axis=1, keepdims=True) + 1e-12)
        clf = Ridge(alpha=alpha, random_state=seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_v)
        results[s] = {
            'r2': float(r2_score(y_v, y_pred)),
            'mse': float(mean_squared_error(y_v, y_pred)),
            'mae': float(mean_absolute_error(y_v, y_pred)),
            'n_train': int(len(y_tr)),
            'n_val': int(len(y_v)),
            'y_mean_train': float(np.mean(y_tr)),
            'y_mean_val': float(np.mean(y_v)),
        }
    return results


def contact_probe(mu_pairs_tr, mu_pairs_v, meta_tr, meta_v,
                  object_names_per_pair, label_root,
                  normalize=False, seed=0):
    """LogReg probe: predict contact (0/1) from frozen μ. Train on tr, eval on val.

    Each pair contributes both μ_a and μ_b vectors per frame, both inheriting
    the SAME contact_pair label (AND of the two sensor contacts), matching
    the spec that within a pair every frame shares one contact state.
    Samples with no label found are skipped (logged)."""
    X_tr_l, y_tr_l, X_v_l, y_v_l = [], [], [], []
    skipped_tr = skipped_v = 0
    missing_pairs = []

    for pk in _PAIRS:
        labels = _load_pair_contact_labels(
            label_root, pk, object_names_per_pair[pk]
        )
        if not labels:
            missing_pairs.append(pk)
            continue

        ma_tr, mb_tr = mu_pairs_tr[pk]
        ma_v, mb_v = mu_pairs_v[pk]
        y_tr = _gather_contact_y(meta_tr[pk], labels, default=-1)
        y_v = _gather_contact_y(meta_v[pk], labels, default=-1)

        m_tr = y_tr >= 0
        m_v = y_v >= 0
        skipped_tr += int((~m_tr).sum())
        skipped_v += int((~m_v).sum())

        # Both μ_a and μ_b at frame t inherit contact_pair[t].
        X_tr_l.append(ma_tr[m_tr].numpy()); y_tr_l.append(y_tr[m_tr])
        X_tr_l.append(mb_tr[m_tr].numpy()); y_tr_l.append(y_tr[m_tr])
        X_v_l.append(ma_v[m_v].numpy()); y_v_l.append(y_v[m_v])
        X_v_l.append(mb_v[m_v].numpy()); y_v_l.append(y_v[m_v])

    if not X_tr_l or not X_v_l:
        return None

    X_tr = np.concatenate(X_tr_l, axis=0)
    y_tr = np.concatenate(y_tr_l, axis=0)
    X_v = np.concatenate(X_v_l, axis=0)
    y_v = np.concatenate(y_v_l, axis=0)

    if normalize:
        X_tr = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-12)
        X_v = X_v / (np.linalg.norm(X_v, axis=1, keepdims=True) + 1e-12)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_v)
    acc = accuracy_score(y_v, y_pred)
    bal_acc = balanced_accuracy_score(y_v, y_pred)
    cm = confusion_matrix(y_v, y_pred, labels=[0, 1])
    return {
        'acc': acc,
        'bal_acc': bal_acc,
        'cm': cm,
        'n_train': int(len(y_tr)),
        'n_val': int(len(y_v)),
        'pos_frac_train': float(y_tr.mean()),
        'pos_frac_val': float(y_v.mean()),
        'skipped_train': skipped_tr,
        'skipped_val': skipped_v,
        'missing_pairs': missing_pairs,
    }


def filter_pairs_by_contact(mu_pairs, meta, object_names_per_pair,
                            label_root, target=1):
    """Return mu_pairs filtered to frames whose contact_pair label == target.

    Both μ_a and μ_b share the same contact_pair label (AND), so the same
    frame mask applies to both sides — paired structure is preserved, which
    is what plot_tsne_paired needs to draw pair links."""
    out = {}
    sizes = {}
    for pk in _PAIRS:
        labels = _load_pair_contact_labels(
            label_root, pk, object_names_per_pair[pk]
        )
        if not labels:
            continue
        ma, mb = mu_pairs[pk]
        y = _gather_contact_y(meta[pk], labels, default=-1)
        keep = y == int(target)
        n_keep = int(keep.sum())
        sizes[pk] = n_keep
        if n_keep == 0:
            continue
        keep_t = torch.from_numpy(keep)
        out[pk] = (ma[keep_t], mb[keep_t])
    return out, sizes


def topk_retrieval(query, gallery, ks=(1, 5, 10)):
    q = F.normalize(query, dim=1)
    g = F.normalize(gallery, dim=1)
    sim = q @ g.t()
    N = sim.shape[0]
    out = {}
    for k in ks:
        topk = sim.topk(k=min(k, N), dim=1).indices
        match = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
        out[k] = match.float().mean().item()
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_tsne(points, perplexity=50, seed=42, metric='cosine'):
    kw = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        early_exaggeration=4,
        metric=metric,
        random_state=seed,
    )
    if 'max_iter' in inspect.signature(TSNE).parameters:
        kw['max_iter'] = 1500
    else:
        kw['n_iter'] = 1500
    coords = TSNE(**kw).fit_transform(points)
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def _subsample(arr, n_max, seed=0):
    if len(arr) <= n_max:
        return arr, np.arange(len(arr))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), n_max, replace=False)
    return arr[idx], idx


def _auto_perplexity(n_total, user_perp=None):
    if n_total < 2:
        raise ValueError('t-SNE needs at least two samples')
    return min(n_total - 1, user_perp if user_perp is not None else min(50, max(5, n_total // 6)))


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

SENSOR_COLOR = {'D': 'tab:blue', 'E': 'tab:orange', 'F': 'tab:green'}
LINK_COLOR = {'de': 'mediumpurple', 'ef': 'gray', 'fd': 'lightcoral'}


def plot_cosine_dist(pairs_data, out_path):
    fig, axes = plt.subplots(1, len(pairs_data), figsize=(5 * len(pairs_data), 4), sharey=False)
    if len(pairs_data) == 1:
        axes = [axes]
    bins = np.linspace(-1, 1, 61)
    for ax, (label, pos, neg, color) in zip(axes, pairs_data):
        ax.hist(neg, bins=bins, alpha=0.55, color='gray', label='neg', density=True)
        ax.hist(pos, bins=bins, alpha=0.75, color=color, label='pos', density=True)
        ax.axvline(pos.mean(), color=color, ls='--', lw=1.5)
        ax.axvline(neg.mean(), color='gray', ls='--', lw=1.5)
        ax.set_title(f"{label}\ngap = {pos.mean() - neg.mean():.3f}")
        ax.set_xlabel('cosine similarity')
        ax.legend(fontsize=8)
    plt.suptitle('μ Cosine: positive vs negative', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_paired(mu_pairs, split_name, out_path, n_max=2000, perplexity=None):
    blocks, info = [], []
    pairs_present = []
    for pk in _PAIRS:
        if pk not in mu_pairs:
            continue
        ma, mb = mu_pairs[pk]
        if len(ma) == 0 or len(mb) == 0:
            continue
        sa, sb = _PAIR_SENSORS[pk]
        a_arr = F.normalize(ma, dim=1).numpy()
        b_arr = F.normalize(mb, dim=1).numpy()
        if n_max > 0:
            a_arr, _ = _subsample(a_arr, n_max, seed=hash(pk + 'a') & 0xFFFF)
            b_arr, _ = _subsample(b_arr, n_max, seed=hash(pk + 'b') & 0xFFFF)
        info.append((sa, pk, len(a_arr)))
        info.append((sb, pk, len(b_arr)))
        blocks.append(a_arr)
        blocks.append(b_arr)
        pairs_present.append(pk)

    if not blocks:
        print(f"  [{split_name}] no frames to plot — skipped {out_path}")
        return

    combined = np.concatenate(blocks, axis=0)
    perp = _auto_perplexity(len(combined), perplexity)
    print(f"  [{split_name}] paired t-SNE on {len(combined)} points (perplexity={perp})")
    coords = _make_tsne(combined, perplexity=perp)

    off = 0
    block_coords = []
    for (_, _, n) in info:
        block_coords.append(coords[off:off + n])
        off += n

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=SENSOR_COLOR[sensor], alpha=0.4, s=12)
    handles = [Line2D([], [], marker='o', ls='', color=c, label=s) for s, c in SENSOR_COLOR.items()]
    ax.legend(handles=handles, fontsize=10)
    ax.set_title(f't-SNE paired [{split_name}] — by modality')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=SENSOR_COLOR[sensor], alpha=0.3, s=10)
    rng = np.random.RandomState(0)
    for i, pk in enumerate(pairs_present):
        pa, pb = block_coords[2 * i], block_coords[2 * i + 1]
        n_links = min(len(pa), len(pb), 100000000)
        if n_links == 0:
            continue
        idx = rng.choice(min(len(pa), len(pb)), n_links, replace=False)
        for k in idx:
            ax.plot([pa[k, 0], pb[k, 0]], [pa[k, 1], pb[k, 1]],
                    c=LINK_COLOR[pk], alpha=0.15, lw=0.5)
    handles = [Line2D([], [], color=c, lw=1.5, label=f'{pk} link')
               for pk, c in LINK_COLOR.items() if pk in pairs_present]
    ax.legend(handles=handles, fontsize=10)
    ax.set_title(f't-SNE paired [{split_name}] — pair links')
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_dynamic_paired(dyn_pair_mus, out_path, title_suffix='',
                             n_max=2000, normalize=True, perplexity=None):
    blocks, info = [], []
    for pk in _PAIRS:
        ma, mb = dyn_pair_mus[pk]
        sa, sb = _PAIR_SENSORS[pk]
        a_arr = F.normalize(ma, dim=1).numpy() if normalize else ma.numpy()
        b_arr = F.normalize(mb, dim=1).numpy() if normalize else mb.numpy()
        if n_max > 0 and len(a_arr) > n_max:
            rng = np.random.RandomState(hash(pk) & 0xFFFF)
            idx = rng.choice(len(a_arr), n_max, replace=False)
            a_arr, b_arr = a_arr[idx], b_arr[idx]
        info.append((sa, pk, len(a_arr)))
        info.append((sb, pk, len(b_arr)))
        blocks.append(a_arr)
        blocks.append(b_arr)

    combined = np.concatenate(blocks, axis=0)
    perp = _auto_perplexity(len(combined), perplexity)
    metric = 'cosine' if normalize else 'euclidean'
    print(f"  [dynamic_paired{title_suffix}] t-SNE on {len(combined)} points (perplexity={perp})")
    coords = _make_tsne(combined, perplexity=perp, metric=metric)

    off = 0
    block_coords = []
    for (_, _, n) in info:
        block_coords.append(coords[off:off + n])
        off += n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    ax = axes[0]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=SENSOR_COLOR[sensor],
                   alpha=0.5, s=20, linewidths=0, rasterized=True)
    handles = [Line2D([], [], marker='o', ls='', color=c, label=s, markersize=8)
               for s, c in SENSOR_COLOR.items()]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc='upper right')
    ax.set_title(f'Dynamic μ — by modality{title_suffix}', fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = axes[1]
    for (sensor, pk, n), pts in zip(info, block_coords):
        ax.scatter(pts[:, 0], pts[:, 1], c=SENSOR_COLOR[sensor],
                   alpha=0.3, s=14, linewidths=0, rasterized=True)
    rng = np.random.RandomState(0)
    for i, pk in enumerate(_PAIRS):
        pa, pb = block_coords[2 * i], block_coords[2 * i + 1]
        n_links = min(len(pa), len(pb), 100000000)
        idx = rng.choice(min(len(pa), len(pb)), n_links, replace=False)
        for k in idx:
            ax.plot([pa[k, 0], pb[k, 0]], [pa[k, 1], pb[k, 1]],
                    c=LINK_COLOR[pk], alpha=0.2, lw=0.5)
    handles = [Line2D([], [], color=c, lw=1.5, label=f'{pk} link') for pk, c in LINK_COLOR.items()]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc='upper right')
    ax.set_title(f'Dynamic μ — pair links{title_suffix}', fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_dynamic_bytask(dyn_mus, task_labels, out_path, n_max=2000,
                             normalize=True, perplexity=None):
    blocks, task_per_point, sensor_per_point = [], [], []
    for s in _SENSORS:
        arr = dyn_mus[s]
        if normalize:
            arr = F.normalize(arr, dim=1)
        arr = arr.numpy()
        labels = np.array(task_labels[s])
        if n_max > 0 and len(arr) > n_max:
            rng = np.random.RandomState(hash('task_' + s) & 0xFFFF)
            idx = rng.choice(len(arr), n_max, replace=False)
            arr, labels = arr[idx], labels[idx]
        blocks.append(arr)
        task_per_point.extend(labels.tolist())
        sensor_per_point.extend([s] * len(arr))

    combined = np.concatenate(blocks, axis=0)
    perp = _auto_perplexity(len(combined), perplexity)
    metric = 'cosine' if normalize else 'euclidean'
    print(f"  [dynamic_bytask] t-SNE on {len(combined)} points (perplexity={perp})")
    coords = _make_tsne(combined, perplexity=perp, metric=metric)

    task_per_point = np.array(task_per_point)
    sensor_per_point = np.array(sensor_per_point)
    unique_tasks = sorted(set(task_per_point))
    task_cmap = dict(zip(unique_tasks,
                         ['tab:red', 'tab:cyan', 'tab:purple', 'tab:olive',
                          'tab:pink', 'tab:brown'][:len(unique_tasks)]))
    sensor_marker = {'D': 'o', 'E': '^', 'F': 's'}

    fig, ax = plt.subplots(figsize=(8, 7))
    for task in unique_tasks:
        for s in _SENSORS:
            mask = (task_per_point == task) & (sensor_per_point == s)
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=task_cmap[task], marker=sensor_marker[s],
                       alpha=0.5, s=20, linewidths=0, rasterized=True,
                       label=f'{task} ({s})')
    ax.set_title('t-SNE dynamic μ — colored by task', fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = []
    for t, c in task_cmap.items():
        handles.append(Line2D([], [], marker='o', ls='', color=c, markersize=8, label=t))
    for s, m in sensor_marker.items():
        handles.append(Line2D([], [], marker=m, ls='', color='gray', markersize=8, label=s))
    ax.legend(handles=handles, fontsize=9, frameon=False, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


def plot_tsne_static_vs_dynamic(static_mu, dyn_mus, out_path, n_max=1500):
    blocks, info = [], []
    for s in _SENSORS:
        st = F.normalize(static_mu[s], dim=1).numpy()
        dy = F.normalize(dyn_mus[s], dim=1).numpy()
        if n_max > 0:
            st, _ = _subsample(st, n_max, seed=hash(f'st_{s}') & 0xFFFF)
            dy, _ = _subsample(dy, n_max, seed=hash(f'dy_{s}') & 0xFFFF)
        blocks.append(st); info.append((s, 'static', len(st)))
        blocks.append(dy); info.append((s, 'dynamic', len(dy)))

    combined = np.concatenate(blocks, axis=0)
    print(f"  [static+dynamic] t-SNE on {len(combined)} points")
    coords = _make_tsne(combined, perplexity=_auto_perplexity(len(combined)))

    fig, ax = plt.subplots(figsize=(9, 8))
    off = 0
    for (s, regime, n) in info:
        pts = coords[off:off + n]; off += n
        marker = 'o' if regime == 'static' else 'x'
        alpha = 0.4 if regime == 'static' else 0.6
        ax.scatter(pts[:, 0], pts[:, 1], c=SENSOR_COLOR[s], alpha=alpha, s=14,
                   marker=marker, label=f"{s} {regime}")
    ax.set_title('t-SNE: static (o) vs dynamic (x) per modality')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Reconstruction visualizations
# ---------------------------------------------------------------------------

@torch.no_grad()
def viz_recon_daimon(val_de, model, device, out_path, n=8):
    n = min(n, len(val_de))
    loader = DataLoader(val_de, batch_size=n, shuffle=False, drop_last=False)
    (xa, xb), _ = next(iter(loader))
    out = model.forward_pair(xa.to(device), xb.to(device), 'de')

    def to_vis(x):
        x = x.cpu().float()
        mn, mx = x.min(), x.max()
        return ((x - mn) / (mx - mn + 1e-8)).clamp(0, 1)

    rows = [
        ('GT Daimon', [to_vis(xa[i]) for i in range(n)]),
        ('Self D(z_D)', [to_vis(out['recon_aa'][i].cpu()) for i in range(n)]),
        ('Cross D(z_E)', [to_vis(out['recon_ba'][i].cpu()) for i in range(n)]),
    ]
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6), squeeze=False)
    for r, (label, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            if img.shape[0] == 3:
                ax.imshow(img.permute(1, 2, 0).numpy())
            else:
                ax.imshow(img.squeeze().numpy(), cmap='gray')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    plt.suptitle('Daimon recon (de_pair val)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


@torch.no_grad()
def viz_recon_eflesh(val_de, model, device, out_path, n=8):
    n = min(n, len(val_de))
    loader = DataLoader(val_de, batch_size=n, shuffle=False, drop_last=False)
    (xa, xb), _ = next(iter(loader))
    out = model.forward_pair(xa.to(device), xb.to(device), 'de')
    gt = xb.cpu().numpy()
    self_r = out['recon_bb'].cpu().numpy()
    cross_r = out['recon_ab'].cpu().numpy()
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 1.2), squeeze=False)
    for i in range(n):
        for j, (arr, ttl) in enumerate([
            (gt[i], 'GT'), (self_r[i], 'Self E(z_E)'), (cross_r[i], 'Cross E(z_D)')
        ]):
            axes[i, j].bar(range(15), arr[i], color='steelblue', alpha=0.8)
            if i == 0:
                axes[i, j].set_title(ttl, fontsize=8)
            axes[i, j].tick_params(labelsize=6)
    plt.suptitle('eFlesh recon (de_pair val, z-scored)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


@torch.no_grad()
def viz_recon_flexi(val_ef, model, device, out_path, n=8):
    n = min(n, len(val_ef))
    loader = DataLoader(val_ef, batch_size=n, shuffle=False, drop_last=False)
    (xe, xf), _ = next(iter(loader))
    out = model.forward_pair(xe.to(device), xf.to(device), 'ef')
    gt = xf.cpu().squeeze(1) if xf.dim() == 4 else xf.cpu()
    self_r = out['recon_bb'].cpu().squeeze(1) if out['recon_bb'].dim() == 4 else out['recon_bb'].cpu()
    cross_r = out['recon_ab'].cpu().squeeze(1) if out['recon_ab'].dim() == 4 else out['recon_ab'].cpu()
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6), squeeze=False)
    for c in range(n):
        for r, (arr, ttl) in enumerate([
            (gt[c].numpy(), 'GT FlexiTac'), (self_r[c].numpy(), 'Self F(z_F)'),
            (cross_r[c].numpy(), 'Cross F(z_E)')
        ]):
            ax = axes[r, c]
            ax.imshow(arr, cmap='viridis')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(ttl, fontsize=8)
    plt.suptitle('FlexiTac recon (ef_pair val)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Eval cross_modal_vae3 with episode_val_split support'
    )
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint_file', default='model.pt',
                        help='Checkpoint filename inside --checkpoint_dir.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset_name', default='dataset_new',
                        help='Entry in datasets_config.yaml')
    parser.add_argument('--objects', nargs='+', default=None)
    parser.add_argument('--dataset_config', help='Override checkpoint data configuration with this YAML')
    parser.add_argument('--data_root', help='Relocate data while preserving saved transforms and split')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--ef_stride', type=int, default=None)
    parser.add_argument('--de_stride', type=int, default=None)
    parser.add_argument('--fd_stride', type=int, default=None)
    parser.add_argument('--dynamic_data_root', nargs='+', default=None,
                        help='Paired-layout dynamic dirs (each has de_pair/ ef_pair/ fd_pair/)')
    parser.add_argument('--dynamic_objects', nargs='+', default=None,
                        help='Object name per dynamic_data_root entry (1:1)')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--max_tsne_points', type=int, default=0,
                        help='0 = use all points')
    parser.add_argument('--tsne_perplexity', type=int, default=None)
    parser.add_argument('--max_mmd_points', type=int, default=2000)
    parser.add_argument('--contact_label_root', default=None,
                        help='Root of per-episode contact label .pt files '
                             '(layout: <root>/<pair_dir>/<obj>/ep****.pt). '
                             'Enables the contact classification probe.')
    parser.add_argument('--contrastive_only', action='store_true',
                        help='Evaluate a CrossModalCLR3 (contrastive-only, no '
                             'decoders) checkpoint instead of CrossModalVAE3. '
                             'Skips the reconstruction error + recon-viz '
                             'sections since the CLR model has no decoders.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.checkpoint_dir, 'eval_new')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load model ──
    model_name = 'cross_modal_clr3' if args.contrastive_only else 'cross_modal_vae3'
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    model = get_model(model_name, model_details).to(args.device)
    checkpoint_file = args.checkpoint_file
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
    ckpt = torch.load(checkpoint_path, map_location=args.device,
                      weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded {checkpoint_path}  (model={model_name})")

    # Saved statistics, objects, strides, and episode split define evaluation.
    if args.dataset_config:
        with open(args.dataset_config) as f:
            dataset_details = yaml.safe_load(f)[args.dataset_name]
    else:
        with open(os.path.join(args.checkpoint_dir, 'dataset_details.yaml')) as f:
            dataset_details = yaml.safe_load(f)
    if args.data_root:
        dataset_details['data_root'] = os.path.abspath(args.data_root)
    elif not os.path.isabs(dataset_details['data_root']):
        dataset_details['data_root'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), dataset_details['data_root'])
    args.objects = args.objects or dataset_details.get('objects') or list(dataset_details.get('val_episode_ids', {}))
    pair_strides = {p: getattr(args, f'{p}_stride') or dataset_details.get('pair_strides', {}).get(p, 1)
                    for p in _PAIRS}
    paired_train, paired_val, _ = get_cross_modal_tri_datasets(
        args.dataset_name, dataset_details, objects=args.objects, pair_strides=pair_strides)
    for split, datasets in [('train', paired_train), ('val', paired_val)]:
        for pair, ds in datasets.items():
            if len(ds) < 2:
                raise ValueError(f'{split}/{pair}: evaluation needs at least two paired frames')
    print('Paired train sizes:', {k: len(v) for k, v in paired_train.items()})
    print('Paired val sizes:', {k: len(v) for k, v in paired_val.items()})

    # ── Extract paired μ (+ meta for contact probe label-alignment) ──
    print('\n=== Extracting paired-static μ ===')
    mu_pairs_tr, mu_pairs_v = {}, {}
    meta_tr, meta_v = {}, {}        # pk -> (obj_id, ep_id, fr_id) tensors
    for pk in _PAIRS:
        ma, mb, oa, ea, fa = extract_pair_mu(
            paired_train[pk], pk, model, args.device, batch_size=args.batch_size, return_meta=True,
        )
        mu_pairs_tr[pk] = (ma, mb)
        meta_tr[pk] = (oa, ea, fa)
        ma, mb, oa, ea, fa = extract_pair_mu(
            paired_val[pk], pk, model, args.device, batch_size=args.batch_size, return_meta=True,
        )
        mu_pairs_v[pk] = (ma, mb)
        meta_v[pk] = (oa, ea, fa)
    object_names_per_pair = {
        pk: list(_unwrap_dataset(paired_train[pk]).object_names) for pk in _PAIRS
    }

    static_mu_per_sensor = {
        'D': mu_pairs_tr['de'][0],
        'E': mu_pairs_tr['de'][1],
        'F': mu_pairs_tr['ef'][1],
    }

    # ── Cosine gap ──
    print('\n=== Cosine gap (val) ===')
    pos_de, neg_de = cosine_pos_neg(*mu_pairs_v['de'])
    pos_ef, neg_ef = cosine_pos_neg(*mu_pairs_v['ef'])
    pos_fd, neg_fd = cosine_pos_neg(*mu_pairs_v['fd'])
    pos_df, neg_df = transitive_df_cosine(
        mu_pairs_v['de'][0], mu_pairs_v['de'][1],
        mu_pairs_v['ef'][0], mu_pairs_v['ef'][1],
    )

    gap_lines = ['Cosine gap (pos_mean - neg_mean) on val:']
    for label, pos, neg in [('DE', pos_de, neg_de), ('EF', pos_ef, neg_ef),
                            ('FD', pos_fd, neg_fd), ('DF_trans', pos_df, neg_df)]:
        gap = pos.mean() - neg.mean()
        gap_lines.append(f"  {label}: pos={pos.mean():.4f}  neg={neg.mean():.4f}  gap={gap:.4f}")
    gap_txt = '\n'.join(gap_lines)
    print(gap_txt)

    pairs_data = [
        ('DE direct', pos_de, neg_de, '#4C72B0'),
        ('EF direct', pos_ef, neg_ef, '#DD8452'),
        ('FD direct', pos_fd, neg_fd, '#55A868'),
        ('DF transitive', pos_df, neg_df, '#C44E52'),
    ]
    plot_cosine_dist(pairs_data, os.path.join(out_dir, 'cosine_dist_val.png'))

    # ── Latent L2 distance on positive pairs (val) ──
    print('\n=== Latent L2 on positive pairs (val) ===')
    l2_lines = ['L2 distance between positive pairs in latent μ space (val):']
    weighted_sum, total_n = 0.0, 0
    for pk in _PAIRS:
        ma, mb = mu_pairs_v[pk]
        mean_l2, std_l2, n = latent_l2_positive(ma, mb)
        l2_lines.append(f"  {pk}: mean={mean_l2:.4f}  std={std_l2:.4f}  (n={n})")
        weighted_sum += mean_l2 * n
        total_n += n
    overall_mean = weighted_sum / total_n
    l2_lines.append(f"  overall (all val pairs): mean L2 = {overall_mean:.4f}  (n={total_n})")
    l2_txt = '\n'.join(l2_lines)
    print(l2_txt)

    # ── Cross-modal retrieval ──
    print('\n=== Cross-modal retrieval (val) ===')
    retr_lines = ['Cross-modal retrieval (paired val, Top-1/5/10):']
    for pk in _PAIRS:
        ma, mb = mu_pairs_v[pk]
        ab = topk_retrieval(ma, mb)
        ba = topk_retrieval(mb, ma)
        retr_lines.append(f"  {pk} a→b: " + '  '.join(f"top{k}={v:.4f}" for k, v in ab.items()))
        retr_lines.append(f"  {pk} b→a: " + '  '.join(f"top{k}={v:.4f}" for k, v in ba.items()))
    retr_txt = '\n'.join(retr_lines)
    print(retr_txt)

    # ── Sensor classification probe (LogReg on μ) ──
    print('\n=== Sensor classification probe (LogReg, train μ → val μ) ===')
    probe_lines = ['Sensor classification probe (LogReg on frozen μ, train→val):']
    for normalize, tag in [(False, 'raw μ'), (True, 'normalized μ')]:
        acc, bal_acc, cm, (n_tr, n_v) = sensor_probe(
            mu_pairs_tr, mu_pairs_v, normalize=normalize, seed=args.seed,
        )
        probe_lines.append(
            f"  [{tag}] accuracy={acc:.4f}  balanced_accuracy={bal_acc:.4f}  "
            f"(n_train={n_tr}, n_val={n_v})"
        )
        probe_lines.append(f"    confusion matrix (rows=true, cols=pred, labels={list(_SENSORS)}):")
        for s, row in zip(_SENSORS, cm):
            probe_lines.append(f"      {s}: " + '  '.join(f"{v:6d}" for v in row))
    probe_txt = '\n'.join(probe_lines)
    print(probe_txt)

    # ── Contact classification probe (LogReg on μ, GT from build_contact_labels.py) ──
    contact_txt = None
    if args.contact_label_root:
        print('\n=== Contact classification probe (LogReg, train μ → val μ) ===')
        contact_lines = [
            'Contact classification probe (LogReg on frozen μ, train→val):',
            f'  label_root: {args.contact_label_root}',
        ]
        any_result = False
        for normalize, tag in [(False, 'raw μ'), (True, 'normalized μ')]:
            res = contact_probe(
                mu_pairs_tr, mu_pairs_v, meta_tr, meta_v,
                object_names_per_pair, args.contact_label_root,
                normalize=normalize, seed=args.seed,
            )
            if res is None:
                contact_lines.append(
                    f"  [{tag}] NO LABELS FOUND under {args.contact_label_root} — skipped"
                )
                continue
            any_result = True
            if res['missing_pairs']:
                contact_lines.append(
                    f"  [{tag}] WARNING missing label dirs for pairs: "
                    f"{res['missing_pairs']}"
                )
            contact_lines.append(
                f"  [{tag}] accuracy={res['acc']:.4f}  "
                f"balanced_accuracy={res['bal_acc']:.4f}  "
                f"(n_train={res['n_train']}, n_val={res['n_val']}, "
                f"pos_frac_tr={res['pos_frac_train']:.3f}, "
                f"pos_frac_v={res['pos_frac_val']:.3f}, "
                f"skipped_no_label tr={res['skipped_train']} v={res['skipped_val']})"
            )
            contact_lines.append("    confusion matrix (rows=true, cols=pred, labels=[0, 1]):")
            for lbl, row in zip([0, 1], res['cm']):
                contact_lines.append(f"      {lbl}: " + '  '.join(f"{v:6d}" for v in row))
        if any_result:
            contact_txt = '\n'.join(contact_lines)
            print(contact_txt)
        else:
            print('\n'.join(contact_lines))
    else:
        print('\n(skip contact probe — pass --contact_label_root to enable)')

    # ── Per-sensor contact-score regression probe (Ridge) ──
    contact_score_txt = None
    if args.contact_label_root:
        print('\n=== Contact-score regression probe (Ridge, per sensor, train μ → val μ) ===')
        cs_lines = [
            'Contact-score regression probe (Ridge on frozen μ, per sensor, train→val):',
            f'  label_root: {args.contact_label_root}',
            '  target: norm_score (per-episode min-max normalized, in [0, 1])',
            '  (each sensor predicts its OWN score — no pair sharing)',
        ]
        any_cs_result = False
        for normalize, tag in [(False, 'raw μ'), (True, 'normalized μ')]:
            res = contact_score_probe(
                mu_pairs_tr, mu_pairs_v, meta_tr, meta_v,
                object_names_per_pair, args.contact_label_root,
                normalize=normalize, seed=args.seed, score_key='norm',
            )
            if not res:
                cs_lines.append(f"  [{tag}] no data — skipped")
                continue
            any_cs_result = True
            for s in _SENSORS:
                if s not in res:
                    cs_lines.append(f"  [{tag}] sensor {s}: no data")
                    continue
                r = res[s]
                cs_lines.append(
                    f"  [{tag}] sensor {s}: R²={r['r2']:+.4f}  "
                    f"MSE={r['mse']:.4f}  MAE={r['mae']:.4f}  "
                    f"(n_train={r['n_train']}, n_val={r['n_val']}, "
                    f"ȳ_tr={r['y_mean_train']:.3f}, ȳ_v={r['y_mean_val']:.3f})"
                )
        if any_cs_result:
            contact_score_txt = '\n'.join(cs_lines)
            print(contact_score_txt)
        else:
            print('\n'.join(cs_lines))

    # ── Reconstruction error (val) — skipped for contrastive-only (no decoders) ──
    recon_txt = None
    if args.contrastive_only:
        print('\n(skip reconstruction L1 error — CLR model has no decoders)')
    else:
        print('\n=== Reconstruction L1 error (val) ===')
        recon_lines = ['L1 reconstruction error (val, z-scored space):']
        for pk in _PAIRS:
            errs = extract_pair_recon(paired_val[pk], pk, model, args.device, batch_size=args.batch_size)
            sa, sb = _PAIR_SENSORS[pk]
            recon_lines.append(f"  {pk}: self_{sa}={errs['self_a']:.4f}  self_{sb}={errs['self_b']:.4f}  "
                               f"cross_{sa}→{sb}={errs['cross_ab']:.4f}  cross_{sb}→{sa}={errs['cross_ba']:.4f}")
        recon_txt = '\n'.join(recon_lines)
        print(recon_txt)

    # ── Save text metrics ──
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(gap_txt + '\n\n')
        f.write(l2_txt + '\n\n')
        f.write(retr_txt + '\n\n')
        f.write(probe_txt + '\n\n')
        if contact_txt is not None:
            f.write(contact_txt + '\n\n')
        if contact_score_txt is not None:
            f.write(contact_score_txt + '\n\n')
        if recon_txt is not None:
            f.write(recon_txt + '\n')

    # ── t-SNE: static train + val ──
    print('\n=== t-SNE plots (static) ===')
    plot_tsne_paired(mu_pairs_tr, 'train', os.path.join(out_dir, 'tsne_static_train.png'),
                     n_max=args.max_tsne_points, perplexity=args.tsne_perplexity)
    plot_tsne_paired(mu_pairs_v, 'val', os.path.join(out_dir, 'tsne_static_val.png'),
                     n_max=args.max_tsne_points, perplexity=args.tsne_perplexity)

    # ── t-SNE: contact / non-contact split (val) ──
    if args.contact_label_root:
        print('\n=== t-SNE plots (val, split by contact label) ===')
        for target, name in [(1, 'contact'), (0, 'noncontact')]:
            filt, sizes = filter_pairs_by_contact(
                mu_pairs_v, meta_v, object_names_per_pair,
                args.contact_label_root, target=target,
            )
            if not filt:
                print(f"  [val_{name}] no frames or no labels — skipped")
                continue
            print(f"  [val_{name}] sizes per pair: {sizes}")
            plot_tsne_paired(
                filt, f'val_{name}',
                os.path.join(out_dir, f'tsne_static_val_{name}.png'),
                n_max=args.max_tsne_points, perplexity=args.tsne_perplexity,
            )

    # ── Reconstruction visualizations — skipped for contrastive-only ──
    if args.contrastive_only:
        print('\n(skip reconstruction visualizations — CLR model has no decoders)')
    else:
        print('\n=== Reconstruction visualizations ===')
        viz_recon_daimon(paired_val['de'], model, args.device,
                         os.path.join(out_dir, 'recon_daimon.png'))
        viz_recon_eflesh(paired_val['de'], model, args.device,
                         os.path.join(out_dir, 'recon_eflesh.png'))
        viz_recon_flexi(paired_val['ef'], model, args.device,
                        os.path.join(out_dir, 'recon_flexi.png'))

    # ── Dynamic evaluation (optional) ──
    if args.dynamic_data_root:
        print('\n=== Dynamic data evaluation ===')
        dyn_roots = args.dynamic_data_root
        if args.dynamic_objects:
            if len(args.dynamic_objects) != len(dyn_roots):
                raise ValueError("--dynamic_objects and --dynamic_data_root must have same length")
            dyn_objs_per_root = [[o] for o in args.dynamic_objects]
        else:
            dyn_objs_per_root = [args.objects for _ in dyn_roots]

        all_dpm = {pk: ([], []) for pk in _PAIRS}
        task_labels = {s: [] for s in _SENSORS}

        for droot, dobjs in zip(dyn_roots, dyn_objs_per_root):
            task_name = dobjs[0] if len(dobjs) == 1 else os.path.basename(droot)
            dyn_details = dict(dataset_details)
            dyn_details['data_root'] = droot
            dyn_details['val_episode_ids'] = []  # Evaluate every dynamic episode
            dyn_train_i, dyn_val_i, _ = get_cross_modal_tri_datasets(
                args.dataset_name, dyn_details, device='cpu', objects=dobjs,
            )
            print(f'  [{task_name}] sizes: {{{", ".join(f"{k}: {len(v)}" for k, v in dyn_train_i.items())}}}')

            for pk in _PAIRS:
                mu_a, mu_b = extract_pair_mu(dyn_train_i[pk], pk, model, args.device, batch_size=args.batch_size)
                all_dpm[pk][0].append(mu_a)
                all_dpm[pk][1].append(mu_b)
                sa, sb = _PAIR_SENSORS[pk]
                task_labels[sa].extend([task_name] * len(mu_a))
                task_labels[sb].extend([task_name] * len(mu_b))

        dpm = {}
        for pk in _PAIRS:
            dpm[pk] = (torch.cat(all_dpm[pk][0]), torch.cat(all_dpm[pk][1]))
        dyn_mus = {
            'D': torch.cat([dpm['de'][0], dpm['fd'][1]], dim=0),
            'E': torch.cat([dpm['de'][1], dpm['ef'][0]], dim=0),
            'F': torch.cat([dpm['ef'][1], dpm['fd'][0]], dim=0),
        }
        print('Dynamic μ sizes:', {s: len(v) for s, v in dyn_mus.items()})

        # MMD static vs dynamic
        print('\n--- MMD static vs dynamic ---')
        mmd_lines = ['MMD² static vs dynamic (lower = better):']
        for s in _SENSORS:
            st = static_mu_per_sensor[s]
            dy = dyn_mus[s]
            n = min(args.max_mmd_points, len(st), len(dy))
            rng = np.random.RandomState(args.seed)
            ids_s = rng.choice(len(st), n, replace=False)
            ids_d = rng.choice(len(dy), n, replace=False)
            mmd = mmd_rbf_multi(st[ids_s], dy[ids_d])
            mmd_lines.append(f"  {s}: MMD² = {mmd:.6f}  (n_static={len(st)}, n_dynamic={len(dy)})")
        mmd_txt = '\n'.join(mmd_lines)
        print(mmd_txt)

        with open(os.path.join(out_dir, 'metrics.txt'), 'a') as f:
            f.write('\n' + mmd_txt + '\n')

        # t-SNE: dynamic paired
        print('\n--- t-SNE (dynamic) ---')
        plot_tsne_dynamic_paired(
            dpm, os.path.join(out_dir, 'tsne_dynamic_paired.png'),
            n_max=args.max_tsne_points, perplexity=args.tsne_perplexity)

        # t-SNE: by task (if multiple roots)
        unique_tasks = set(sum((task_labels[s] for s in _SENSORS), []))
        if len(unique_tasks) > 1:
            plot_tsne_dynamic_bytask(
                dyn_mus, task_labels,
                os.path.join(out_dir, 'tsne_dynamic_bytask.png'),
                n_max=args.max_tsne_points, perplexity=args.tsne_perplexity)

        # t-SNE: static vs dynamic overlay
        plot_tsne_static_vs_dynamic(
            static_mu_per_sensor, dyn_mus,
            os.path.join(out_dir, 'tsne_static_vs_dynamic.png'),
            n_max=min(args.max_tsne_points, 1500) if args.max_tsne_points > 0 else 1500,
        )

    print(f"\nDone. All outputs under: {out_dir}/")


if __name__ == '__main__':
    main()
