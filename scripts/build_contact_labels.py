"""
Build per-episode contact ground-truth labels from tactile .pt frames.

For each (pair, object, episode) under <data_root>/<pair>/<sensor>/<obj>/ep*_fr*.pt:
  1. Per-frame contact score per sensor:
        score_t = mean_abs(x_t - baseline),
        baseline = mean of the first --baseline_n frames of that episode.
  2. Per-episode min-max normalize each sensor's score -> [0, 1].
  3. Threshold per (episode, sensor). Default = fixed `--threshold` (0.10).
     Rationale: in this dataset every non-zero state is contact (the 0->1
     rising edge is already pressing), so we want any normalized score
     clearly above the noise floor to count as contact. A small fixed
     threshold is more honest about that than Otsu, which finds the split
     between the low and high modes and tends to push the rising edge of
     the envelope into the non-contact class.
     Optional: --use_otsu falls back to Otsu's method on the normalized
     score, clipped to [--min_thresh, --max_thresh].
  4. contact_a = norm_a > thresh_a, contact_b = norm_b > thresh_b
     contact_pair = contact_a AND contact_b  (conservative).
     Rationale: eflesh decays fast after release while flexitac decays
     slowly, so any mean / OR / soft signal stays high too long. AND only
     marks contact while BOTH sensors are pressing, and matches the spec
     that within a pair every frame must share the same contact state.
  5. Optional 1D morphological close (fill short 0-gaps) + open (drop short
     1-spikes) for denoising.
  6. Save one .pt per episode at:
        <out_root>/<pair>/<object>/ep****.pt
     with keys:
        episode, object, pair, sensor_a, sensor_b,
        frames, scores_a, scores_b, norm_a, norm_b,
        thresh_a, thresh_b,
        contact_a, contact_b, contact_pair, baseline_n
  7. (--visualize) per-episode plot at
        <plot_dir>/<pair>/<object>/ep****.png

Notes:
  - Does NOT modify or copy any original tactile .pt; only writes a small
    labels file per episode (a few hundred bytes).
  - Eval (scripts/evaluate.py --contact_label_root ...)
    loads these labels and matches them to encoder μ samples via
    (object_id, episode_id, frame_id), so the probe sees the SAME frames
    the model was evaluated on.

Usage:
    # default: fixed threshold 0.1 on per-episode min-max normalized score
    python scripts/build_contact_labels.py \
        --data_root policy_learning/data/latent_data_new_processed_concat \
        --out_root contact_labels/dataset_new_concat \
        --visualize
    OBJECTS="circle cylinder cylinder_vertical ellipse large_ball plane plane_horizontal small_ball square triangle"

for OBJ in $OBJECTS; do
    echo "Processing $OBJ..."

    python scripts/build_contact_labels.py \
        --data_root policy_learning/data/latent_data_new_processed_concat \
        --out_root contact_labels/dataset_new_concat \
        --objects "$OBJ" \
        --visualize
done
        

    # tweak the threshold
    python scripts/build_contact_labels.py ... --threshold 0.08

    # use Otsu instead (auto per-episode/sensor)
    python scripts/build_contact_labels.py ... --use_otsu
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


PAIR_SENSORS = {
    'de_pair': ('daimon', 'eflesh'),
    'ef_pair': ('eflesh', 'flexi'),
    'fd_pair': ('flexi', 'daimon'),
}

_FNAME_RE = re.compile(r'ep(\d+)_fr(\d+)\.pt')


def _parse_ep_fr(name):
    m = _FNAME_RE.match(os.path.basename(name))
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _load_raw(path):
    """Load a tactile frame .pt as a float tensor (handles dict-wrapped files)."""
    x = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(x, dict):
        for k in ('image', 'magnetic', 'flexi', 'flexitac', 'tactile',
                  'data', 'obs', 'x'):
            if k in x:
                x = x[k]
                break
    return torch.as_tensor(x).float()


def episode_frames(sensor_dir, episode):
    files = glob.glob(os.path.join(sensor_dir, f'ep{episode:04d}_fr*.pt'))
    items = []
    for f in files:
        p = _parse_ep_fr(f)
        if p is None:
            continue
        items.append((p[1], f))
    items.sort(key=lambda x: x[0])
    return items


def contact_score_seq(sensor_dir, episode, baseline_n=1):
    """Return (frames, raw_scores) for one (sensor, episode)."""
    items = episode_frames(sensor_dir, episode)
    if not items:
        return None, None
    xs = [_load_raw(p) for _, p in items]
    baseline_n = max(1, min(baseline_n, len(xs)))
    base = torch.stack(xs[:baseline_n], dim=0).mean(dim=0)
    scores = np.array([(x - base).abs().mean().item() for x in xs],
                      dtype=np.float32)
    frames = np.array([fr for fr, _ in items], dtype=np.int64)
    return frames, scores


def minmax_norm(a, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < eps:
        return np.zeros_like(a, dtype=np.float64)
    return (a - lo) / (hi - lo)


def otsu_threshold(values_in_unit_range, n_bins=64):
    """Otsu's method for 1D values in [0,1]. Returns threshold in (0,1)."""
    v = np.asarray(values_in_unit_range, dtype=np.float64)
    v = np.clip(v, 0.0, 1.0)
    hist, edges = np.histogram(v, bins=n_bins, range=(0.0, 1.0))
    p = hist / max(1, hist.sum())
    centers = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom = np.where(denom < 1e-12, np.nan, denom)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b2))
    return float(centers[k])


def _runs(mask):
    """Yield (start, end_exclusive, value) for a 1D bool array."""
    if len(mask) == 0:
        return
    cur = bool(mask[0])
    s = 0
    for i in range(1, len(mask)):
        if bool(mask[i]) != cur:
            yield s, i, cur
            cur = bool(mask[i])
            s = i
    yield s, len(mask), cur


def morph_close_open(mask, min_gap=2, min_run=2):
    """Fill interior 0-gaps shorter than `min_gap`, then drop 1-runs shorter
    than `min_run`. Border 0-runs are NEVER filled (we don't want to mark the
    pre-touch / post-release tails as contact)."""
    out = np.asarray(mask, dtype=bool).copy()
    if len(out) == 0:
        return out
    for s, e, v in list(_runs(out)):
        if (not v) and (e - s) < min_gap and s > 0 and e < len(out):
            out[s:e] = True
    for s, e, v in list(_runs(out)):
        if v and (e - s) < min_run:
            out[s:e] = False
    return out


def _episode_threshold(norm_scores, mode, fixed, min_thresh, max_thresh):
    """Choose a threshold for one (episode, sensor) normalized score array."""
    if mode == 'fixed':
        return float(fixed)
    if mode == 'otsu':
        return float(np.clip(otsu_threshold(norm_scores), min_thresh, max_thresh))
    raise ValueError(f"unknown threshold mode '{mode}'")


def build_one_episode(data_root, pair, obj, episode,
                      baseline_n=1, threshold_mode='fixed',
                      fixed_threshold=0.1,
                      min_thresh=0.15, max_thresh=0.6,
                      smooth_min_gap=2, smooth_min_run=2):
    sa, sb = PAIR_SENSORS[pair]
    sa_dir = os.path.join(data_root, pair, sa, obj)
    sb_dir = os.path.join(data_root, pair, sb, obj)
    fr_a, sc_a = contact_score_seq(sa_dir, episode, baseline_n)
    fr_b, sc_b = contact_score_seq(sb_dir, episode, baseline_n)
    if fr_a is None or fr_b is None:
        return None

    common = np.intersect1d(fr_a, fr_b)
    if len(common) == 0:
        return None
    sel_a = np.isin(fr_a, common)
    sel_b = np.isin(fr_b, common)
    fr_a, sc_a = fr_a[sel_a], sc_a[sel_a]
    fr_b, sc_b = fr_b[sel_b], sc_b[sel_b]
    # both already sorted by frame because items are sorted in episode_frames

    norm_a = minmax_norm(sc_a)
    norm_b = minmax_norm(sc_b)
    th_a = _episode_threshold(norm_a, threshold_mode, fixed_threshold,
                              min_thresh, max_thresh)
    th_b = _episode_threshold(norm_b, threshold_mode, fixed_threshold,
                              min_thresh, max_thresh)

    contact_a = norm_a > th_a
    contact_b = norm_b > th_b

    # Asymmetric onset/offset state machine:
    #   OFF → ON  : both sensors must exceed threshold (AND, follow the
    #               slower-rising sensor)
    #   ON  → OFF : either sensor drops below threshold (OR-exit, follow
    #               the faster-falling sensor — e.g. eflesh drops quickly
    #               after release while flexitac decays slowly, so eflesh
    #               triggers the exit)
    n = len(contact_a)
    contact_pair_raw = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        if not state:
            if contact_a[t] and contact_b[t]:
                state = True
        else:
            if (not contact_a[t]) or (not contact_b[t]):
                state = False
        contact_pair_raw[t] = state

    contact_pair = morph_close_open(
        contact_pair_raw, min_gap=smooth_min_gap, min_run=smooth_min_run,
    )

    return {
        'episode': int(episode),
        'object': obj,
        'pair': pair,
        'sensor_a': sa,
        'sensor_b': sb,
        'frames': common.astype(np.int64),
        'scores_a': sc_a.astype(np.float32),
        'scores_b': sc_b.astype(np.float32),
        'norm_a': norm_a.astype(np.float32),
        'norm_b': norm_b.astype(np.float32),
        'thresh_a': th_a,
        'thresh_b': th_b,
        'threshold_mode': threshold_mode,
        'contact_a': contact_a.astype(np.uint8),
        'contact_b': contact_b.astype(np.uint8),
        'contact_pair': contact_pair.astype(np.uint8),
        'baseline_n': int(baseline_n),
    }


def plot_episode(d, out_path):
    fr = d['frames']
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(fr, d['norm_a'], color='#1f77b4', lw=1.8,
            label=f"{d['sensor_a']} (a)")
    ax.plot(fr, d['norm_b'], color='#ff7f0e', lw=1.8,
            label=f"{d['sensor_b']} (b)")
    ax.axhline(d['thresh_a'], ls='--', color='#1f77b4', lw=1.0,
               label=f"th_a = {d['thresh_a']:.2f}")
    ax.axhline(d['thresh_b'], ls='--', color='#ff7f0e', lw=1.0,
               label=f"th_b = {d['thresh_b']:.2f}")
    cp = d['contact_pair'].astype(bool)

    # Shade contact frames as frame-centered bins.
    first_label = True
    for s, e, v in _runs(cp):
        if not v:
            continue

        x0 = fr[s] - 0.5
        x1 = fr[e - 1] + 0.5

        ax.axvspan(
            x0, x1,
            color='gray',
            alpha=0.35,
            label='contact_pair' if first_label else None
        )
    first_label = False

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(int(fr.min()) - 0.5, int(fr.max()) + 0.5)
    ax.set_xlabel('frame')
    ax.set_ylabel('normalized score')
    ax.set_title(
        f"{d['pair']} | {d['object']} | ep{d['episode']:04d} | "
        f"contact={int(cp.sum())}/{len(cp)}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def list_episodes(data_root, pair, obj):
    sa, _ = PAIR_SENSORS[pair]
    pat = os.path.join(data_root, pair, sa, obj, 'ep*_fr*.pt')
    eps = set()
    for f in glob.glob(pat):
        p = _parse_ep_fr(f)
        if p is not None:
            eps.add(p[0])
    return sorted(eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True,
                        help='Tactile data root (has <pair>/<sensor>/<obj>/ep*_fr*.pt)')
    parser.add_argument('--out_root', required=True,
                        help='Output root for label .pt files')
    parser.add_argument('--objects', nargs='+', default=['small_ball'])
    parser.add_argument('--pairs', nargs='+', default=list(PAIR_SENSORS.keys()))
    parser.add_argument('--baseline_n', type=int, default=1,
                        help='# of first frames used as baseline (default 1)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Fixed threshold on per-episode min-max normalized '
                             'score (default 0.1). Used when --use_otsu is not set.')
    parser.add_argument('--use_otsu', action='store_true',
                        help='Use Otsu per (episode, sensor) instead of fixed '
                             '--threshold. Otsu finds the cut between low/high '
                             'modes and may push rising-edge frames into the '
                             'non-contact class; use only if you want that.')
    parser.add_argument('--min_thresh', type=float, default=0.15,
                        help='Lower clip for Otsu-estimated threshold '
                             '(ignored unless --use_otsu).')
    parser.add_argument('--max_thresh', type=float, default=0.6,
                        help='Upper clip for Otsu-estimated threshold '
                             '(ignored unless --use_otsu).')
    parser.add_argument('--smooth_min_gap', type=int, default=2,
                        help='close interior 0-gaps shorter than this')
    parser.add_argument('--smooth_min_run', type=int, default=2,
                        help='drop 1-runs shorter than this')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--plot_dir', default=None,
                        help='Plot output dir (default <out_root>/_plots)')
    args = parser.parse_args()

    plot_dir = args.plot_dir or os.path.join(args.out_root, '_plots')
    threshold_mode = 'otsu' if args.use_otsu else 'fixed'
    print(f"Threshold mode: {threshold_mode}"
          + (f" (value={args.threshold})" if threshold_mode == 'fixed'
             else f" (clip=[{args.min_thresh}, {args.max_thresh}])"))

    total_eps = total_contact = total_frames = 0
    skipped = 0

    for pair in args.pairs:
        if pair not in PAIR_SENSORS:
            print(f'[skip] unknown pair: {pair}')
            continue
        for obj in args.objects:
            eps = list_episodes(args.data_root, pair, obj)
            if not eps:
                print(f'[skip] no episodes for {pair}/{obj}')
                continue
            out_dir = os.path.join(args.out_root, pair, obj)
            os.makedirs(out_dir, exist_ok=True)
            if args.visualize:
                p_dir = os.path.join(plot_dir, pair, obj)
                os.makedirs(p_dir, exist_ok=True)
            print(f'=== {pair}/{obj}: {len(eps)} episodes ===')
            for ep in tqdm(eps, desc=f'{pair}/{obj}'):
                d = build_one_episode(
                    args.data_root, pair, obj, ep,
                    baseline_n=args.baseline_n,
                    threshold_mode=threshold_mode,
                    fixed_threshold=args.threshold,
                    min_thresh=args.min_thresh, max_thresh=args.max_thresh,
                    smooth_min_gap=args.smooth_min_gap,
                    smooth_min_run=args.smooth_min_run,
                )
                if d is None:
                    skipped += 1
                    continue
                torch.save(d, os.path.join(out_dir, f'ep{ep:04d}.pt'))
                if args.visualize:
                    plot_episode(d, os.path.join(plot_dir, pair, obj,
                                                 f'ep{ep:04d}.png'))
                total_eps += 1
                total_contact += int(d['contact_pair'].sum())
                total_frames += len(d['frames'])

    print()
    if total_frames > 0:
        print(f"Saved labels for {total_eps} episodes -> {args.out_root}/")
        print(f"Contact fraction (AND): {total_contact}/{total_frames} "
              f"= {total_contact / total_frames:.3f}")
    if skipped:
        print(f"Skipped {skipped} episodes (missing frames in one of the sensors).")
    if args.visualize:
        print(f"Plots -> {plot_dir}/")


if __name__ == '__main__':
    main()
