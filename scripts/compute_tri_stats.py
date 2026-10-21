"""Compute per-sensor normalization statistics for the tri-modal dataset.

Reads:
    latent_data_processed/{de_pair, fd_pair}/daimon/*.pt   (raw tensor (3, H, W))
    latent_data_processed/{de_pair, ef_pair}/eflesh/*.pt   (raw tensor (15,))
    latent_data_processed/{ef_pair, fd_pair}/flexi/*.pt    (raw tensor (12, 16))

Outputs YAML-style stats to stdout. Paste them into
``config/datasets_config.yaml`` under ``dataset_new``.

Usage:
    python scripts/compute_tri_stats.py
    python scripts/compute_tri_stats.py --data_root latent_data_processed
"""

import argparse
import glob
import os
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


def _iter_files(roots: Iterable[str], sensor: str):
    for r in roots:
        d = os.path.join(r, sensor)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, '**', '*.pt'), recursive=True)):
            yield p


def _load_tensor(path: str) -> torch.Tensor:
    x = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(x, dict):
        if 'image' in x: x = x['image']
        elif 'magnetic' in x: x = x['magnetic']
        elif 'flexi' in x: x = x['flexi']
        elif 'flexitac' in x: x = x['flexitac']
        else: raise ValueError(f"unknown keys {list(x.keys())} in {path}")
    return x.float()


def daimon_stats(roots):
    """Per-channel mean/std over (3, H, W) Daimon tensors."""
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    n_pix = 0
    files = list(_iter_files(roots, 'daimon'))
    for p in tqdm(files, desc='daimon'):
        t = _load_tensor(p).double().numpy()  # (3, H, W)
        sum_c += t.sum(axis=(1, 2))
        sumsq_c += (t * t).sum(axis=(1, 2))
        n_pix += t.shape[1] * t.shape[2]
    if n_pix == 0:
        return None
    mean = sum_c / n_pix
    var = sumsq_c / n_pix - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist(), len(files)


def eflesh_stats(roots):
    """Per-dim mean/std over (15,) eFlesh vectors."""
    files = list(_iter_files(roots, 'eflesh'))
    arr = np.zeros((len(files), 15), dtype=np.float64)
    for i, p in enumerate(tqdm(files, desc='eflesh')):
        arr[i] = _load_tensor(p).double().numpy().reshape(-1)
    if len(files) == 0:
        return None
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist(), len(files)


def flexi_stats(roots):
    """Scalar mean/std over (12, 16) FlexiTac tensors (treat as one channel)."""
    s = 0.0
    sq = 0.0
    n = 0
    files = list(_iter_files(roots, 'flexi'))
    for p in tqdm(files, desc='flexi'):
        t = _load_tensor(p).double().numpy()
        s += t.sum()
        sq += (t * t).sum()
        n += t.size
    if n == 0:
        return None
    mean = s / n
    var = sq / n - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std, len(files)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default='latent_data_processed',
                   help='Root containing de_pair/, ef_pair/, fd_pair/.')
    return p.parse_args()


def main():
    args = parse_args()
    root = args.data_root
    de = os.path.join(root, 'de_pair')
    ef = os.path.join(root, 'ef_pair')
    fd = os.path.join(root, 'fd_pair')

    print('=' * 60)
    print('Tri-modal normalization stats')
    print('=' * 60)
    print(f'  Data root: {root}')
    print()

    daimon = daimon_stats([de, fd])
    eflesh = eflesh_stats([de, ef])
    flexi = flexi_stats([ef, fd])

    print()
    print('Paste under dataset_new in config/datasets_config.yaml:')
    print('-' * 60)

    if daimon is not None:
        m, s, n = daimon
        print(f'  # daimon: {n} files')
        print(f'  daimon_stats:')
        print(f'    mean: [{m[0]:.6f}, {m[1]:.6f}, {m[2]:.6f}]')
        print(f'    std:  [{s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f}]')

    if eflesh is not None:
        m, s, n = eflesh
        print(f'  # eflesh: {n} files')
        print(f'  eflesh_stats:')
        print(f'    mean: [' + ', '.join(f'{x:.6f}' for x in m) + ']')
        print(f'    std:  [' + ', '.join(f'{x:.6f}' for x in s) + ']')

    if flexi is not None:
        m, s, n = flexi
        print(f'  # flexi: {n} files')
        print(f'  flexitac_stats:')
        print(f'    mean: {m:.6f}')
        print(f'    std:  {s:.6f}')


if __name__ == '__main__':
    main()
