"""Joint alignment training for the three paired tactile sensors."""
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from joint_embedding_learning.utils.models_setup import get_model
from joint_embedding_learning.utils.utils import load_optimizer, get_loss, save_model

def compute_embedding_metrics(z_i, z_j):
    """Compute embedding-quality metrics on a batch of paired projections.

    Args:
        z_i, z_j: (B, D) projection-head outputs from the two modalities. Diagonal pairs
            (z_i[k], z_j[k]) are positives; off-diagonal cross-modal pairs are negatives.

    Returns:
        dict with pos/neg cosine sim mean+std, gap, and alignment (||z_i - z_j||^2 mean).
    """
    z_i_n = F.normalize(z_i, dim=1)
    z_j_n = F.normalize(z_j, dim=1)
    sim = z_i_n @ z_j_n.t()                                # (B, B)
    B = sim.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    pos = sim[eye]                                         # (B,)
    neg = sim[~eye]                                        # (B*(B-1),)
    alignment = (z_i_n - z_j_n).pow(2).sum(dim=1).mean()
    pos_std = pos.std(unbiased=False) if B > 1 else torch.zeros((), device=sim.device)
    neg_std = neg.std(unbiased=False) if neg.numel() > 1 else torch.zeros((), device=sim.device)
    return {
        'pos_cos_mean': pos.mean().item(),
        'pos_cos_std' : pos_std.item(),
        'neg_cos_mean': neg.mean().item() if neg.numel() else 0.0,
        'neg_cos_std' : neg_std.item(),
        'gap'         : (pos.mean() - neg.mean()).item() if neg.numel() else 0.0,
        'alignment'   : alignment.item(),
    }


def _kl_beta(args):
    """Linear warm-up: 0 -> beta_max over kl_warmup_epochs, then clamped at beta_max."""
    beta_max = float(getattr(args, 'beta_max', 0.0))
    warmup = max(int(getattr(args, 'kl_warmup_epochs', 0)), 1)
    frac = min(1.0, float(args.current_epoch) / warmup)
    return beta_max * frac


def _infonce_alpha(args):
    """Delayed linear warm-up for InfoNCE: 0 until start epoch, then 0 -> 1."""
    start = int(getattr(args, 'infonce_warmup_start', 0))
    warmup = int(getattr(args, 'infonce_warmup_epochs', 0))
    epoch = args.current_epoch
    if epoch < start:
        return 0.0
    return 1.0 if warmup <= 0 else min(1.0, (epoch - start) / warmup)


def _transitive_df_metric(out_de, out_ef):
    """Indirect D-F alignment proxy.

    For each ef_pair sample, find the nearest de_pair sample by cosine of mu_E
    (the bridge sensor present in both pairs), then align the corresponding
    mu_F (from ef_pair) with mu_D (from de_pair).

    Returns the diagonal-vs-offdiagonal cosine gap on the implied (mu_F, mu_D)
    correspondence; > 0 means transitive D-F alignment is emerging.
    """
    mu_E_de = out_de["mu_b"]   # de pair: a=D, b=E
    mu_D_de = out_de["mu_a"]
    mu_E_ef = out_ef["mu_a"]   # ef pair: a=E, b=F
    mu_F_ef = out_ef["mu_b"]

    a = F.normalize(mu_E_ef, dim=1)
    b = F.normalize(mu_E_de, dim=1)
    sim_E = a @ b.t()  # (B, B)
    nn_idx = sim_E.argmax(dim=1)  # for each ef sample, idx of most similar de sample
    mu_D_match = mu_D_de[nn_idx]

    z_F = F.normalize(mu_F_ef, dim=1)
    z_D = F.normalize(mu_D_match, dim=1)
    sim = z_F @ z_D.t()
    B = sim.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    pos = sim[eye]
    neg = sim[~eye]
    return {
        'df_pos_cos_mean': pos.mean().item(),
        'df_neg_cos_mean': neg.mean().item() if neg.numel() else 0.0,
        'df_gap'         : (pos.mean() - neg.mean()).item() if neg.numel() else 0.0,
    }


def _tri_log_metrics(prefix, outs, info):
    """Build wandb log dict with cosine metrics per pair grouped under shared prefix.

    For each metric kind in {cos_pos, cos_neg, cos_gap, alignment}, all four pair
    series ('de', 'ef', 'fd', 'df_transitive') are emitted under the same
    sub-prefix so wandb plots them on one chart automatically:

        {prefix}/cos_pos/de, {prefix}/cos_pos/ef, {prefix}/cos_pos/fd,
        {prefix}/cos_pos/df_transitive
        {prefix}/cos_neg/...
        {prefix}/cos_gap/...
        {prefix}/alignment/...

    Per-pair losses live under "Loss_..." groups (caller builds them).
    """
    log = {}
    for p in ('de', 'ef', 'fd'):
        m = compute_embedding_metrics(outs[p]["mu_a"], outs[p]["mu_b"])
        log[f"{prefix}/cos_pos/{p}"] = m['pos_cos_mean']
        log[f"{prefix}/cos_neg/{p}"] = m['neg_cos_mean']
        log[f"{prefix}/cos_gap/{p}"] = m['gap']
        log[f"{prefix}/alignment/{p}"] = m['alignment']

    tm = _transitive_df_metric(outs['de'], outs['ef'])
    log[f"{prefix}/cos_pos/df_transitive"] = tm['df_pos_cos_mean']
    log[f"{prefix}/cos_neg/df_transitive"] = tm['df_neg_cos_mean']
    log[f"{prefix}/cos_gap/df_transitive"] = tm['df_gap']

    log[f"{prefix}/cos_gap_avg_direct"] = (
        log[f"{prefix}/cos_gap/de"] + log[f"{prefix}/cos_gap/ef"]
        + log[f"{prefix}/cos_gap/fd"]
    ) / 3.0
    return log


def _tri_zip_loaders(loaders):
    """Restart shorter loaders without caching an epoch of tensor batches."""
    if set(loaders) != {'de', 'ef', 'fd'}:
        raise ValueError("All three paired loaders (de, ef, fd) are required")
    empty = [key for key, loader in loaders.items() if len(loader) == 0]
    if empty:
        raise ValueError(f"Empty paired loaders: {empty}; check data, splits, and batch_size")
    iters = {key: iter(loader) for key, loader in loaders.items()}
    for _ in range(max(map(len, loaders.values()))):
        batches = {}
        for key, loader in loaders.items():
            try:
                batches[key] = next(iters[key])
            except StopIteration:
                iters[key] = iter(loader)
                batches[key] = next(iters[key])
        yield batches


def _run_epoch(args, loaders, model, criterion, logger, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batches in tqdm(_tri_zip_loaders(loaders), total=max(map(len, loaders.values())), leave=False):
            targets = {p: tuple(x.to(args.device) for x in batch[0]) for p, batch in batches.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = {p: model.forward_pair(*targets[p], p) for p in ('de', 'ef', 'fd')}
                info = criterion(outputs, targets, beta=_kl_beta(args), alpha=_infonce_alpha(args))
                loss = info['total']
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {args.current_epoch}")
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            prefix = 'train' if training else 'val'
            with torch.no_grad():
                logs = {f'Loss_{prefix}/{k}': float(v.detach()) if torch.is_tensor(v) else float(v)
                        for k, v in info.items()}
                logs.update(_tri_log_metrics(f'Metric_{prefix}', outputs, info))
                logs['Epoch'] = args.current_epoch
                logger.log(logs)
            total += loss.item()
            count += 1
    return total / count


def train_tri(train_datasets, validation_datasets, dataset_name, dataset_details,
              model_name, model_details, wandb, run_name='', device='cuda:0',
              resume_from=None):
    """Train all three pairs and save strict, checkpoint-compatible state dictionaries.

    resume_from warm-starts weights only; optimizer and epoch state start anew.
    """
    args = SimpleNamespace(**model_details)
    args.device = device
    if args.batch_size < 2:
        raise ValueError('Training batch_size must be at least 2 for contrastive learning')
    if args.epochs <= args.start_epoch:
        raise ValueError('epochs must exceed start_epoch')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    workers = int(getattr(args, 'num_workers', 0))
    loader_options = dict(num_workers=workers, pin_memory=device.startswith('cuda'))
    if workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    def loaders(datasets, training):
        if set(datasets) != {'de', 'ef', 'fd'}:
            raise ValueError('Dataset must contain de, ef, and fd pairs')
        for key, ds in datasets.items():
            required = args.batch_size if training else 1
            if len(ds) < required:
                raise ValueError(f"{key}: {len(ds)} samples; need at least {required} for "
                                 f"{'training' if training else 'validation'}. Check paths and episode split.")
        return {p: DataLoader(ds, batch_size=args.batch_size, shuffle=training,
                              drop_last=training, **loader_options) for p, ds in datasets.items()}
    train_loaders = loaders(train_datasets, True)
    val_loaders = {name: loaders(ds, False) for name, ds in validation_datasets.items() if ds}
    model = get_model(model_name, model_details).to(device)
    if resume_from:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state.get('state_dict', state), strict=True)
    optimizer, _, _ = load_optimizer(args, model)
    criterion = get_loss(args, model_name).to(device)
    scaler = torch.amp.GradScaler('cuda') if getattr(args, 'amp', False) and device.startswith('cuda') else None
    path = None
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc='Epochs'):
        args.current_epoch = epoch
        train_loss = _run_epoch(args, train_loaders, model, criterion, wandb, optimizer, scaler)
        wandb.log({'Loss_train/Model_Loss': train_loss, 'Epoch': epoch})
        for name, split_loaders in val_loaders.items():
            val_loss = _run_epoch(args, split_loaders, model, criterion, wandb)
            wandb.log({f'Loss_{name}/Model_Loss': val_loss, 'Epoch': epoch})
        print(f'Epoch {epoch}: training loss {train_loss:.6f}')
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            tag = f'ep{epoch}' if epoch >= 100 and epoch % 50 == 0 else None
            path = save_model(model, dataset_name, dataset_details['dataset_type'], model_name,
                              dataset_details, model_details, run_name, tag=tag,
                              checkpoint_root=getattr(args, 'checkpoint_root', None))
    return path
