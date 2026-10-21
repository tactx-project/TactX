"""
Convert teleop HDF5 episodes to 16-d latent μ using a trained
cross_modal_vae3 checkpoint with saved normalization statistics.

Reads each HDF5 episode, encodes the per-sensor tactile streams to the
shared 16-d latent, and writes a new HDF5 with observation.tac0/tac1
replaced by latents. Everything else is copied verbatim.

Per-sensor I/O schemas (matches the dataset documented demonstration layout):

    Sensor D (Daimon) — vision-based, raw 5-channel array:
        in : observation.tac0/tac1   (N, 384000) float32
             (= flattened (N, 240, 320, 5) raw daimon: depth + 4 derived chans)
        ops: reshape (N, 240, 320, 5) → pick channels [0, 3, 4]
             = [depth, shear_x, shear_y] → permute (N, 3, H, W) → resize 224 →
             z-score with dataset_details daimon_stats → model._enc(x, 'D')
        out: observation.tac0/tac1   (N, 16) float32

        IMPORTANT: observation.image.tac0/tac1 (uint8 RGB frames decoded from
        tac*.mp4) is a *visualization* of the sensor and is NOT the right input
        — the encoder was trained on [depth, shear_x, shear_y] (see
        preprocess.py::daimon_to_chw, channel indices 0/3/4).

    Sensor E (eFlesh) — magnetic vector:
        in : observation.tac0/tac1         (N, 15) float32
        ops: z-score with eflesh_stats → model._enc(x, 'E')
        out: observation.tac0/tac1         (N, 16) float32

    Sensor F (FlexiTac) — resistive matrix:
        in : observation.tac0/tac1         (N, 192) float32
        ops: reshape (N, 12, 16) → z-score with flexitac_stats →
             model._enc(x, 'F')
        out: observation.tac0/tac1         (N, 16) float32

Optional --inference_transform applies a deterministic fliplr/flipud/rot180
to the raw input before encoding. Useful if the teleop data was collected
in a different orientation than the training data baked into the
dataset_new per-pair transforms. Defaults to 'none'.

Usage:
    python scripts/convert_tactile_to_latent.py \
        --checkpoint_dir checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_<name>_<TS> \
        --sensor E \
        --input_dir  policy_learning/data/recordings/processed/eflesh_eflesh_object_pnp \
        --output_dir policy_learning/data/recordings/processed/eflesh_eflesh_object_pnp_latent
"""

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joint_embedding_learning.utils.models_setup import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_transform(x, mode):
    """fliplr / flipud / rot180 on the last 1-2 dims (negative indices)."""
    if mode in (None, 'none'):
        return x
    if x.dim() < 2 and mode in ('flipud', 'rot180'):
        return torch.flip(x, dims=[-1]) if mode == 'fliplr' else x
    if mode == 'fliplr':
        return torch.flip(x, dims=[-1])
    if mode == 'flipud':
        return torch.flip(x, dims=[-2])
    if mode == 'rot180':
        return torch.flip(x, dims=[-2, -1])
    raise ValueError(f"Unknown transform: {mode}")


def _zscore(x, mean, std):
    """Channel-wise z-score. mean/std broadcast over leading channel dim."""
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(std, dtype=torch.float32)
    # Broadcast: x shape (B, C, H, W) -> reshape stats to (C, 1, 1); (B, 15) -> (15,) is fine.
    if x.dim() == 4 and m.dim() == 1 and m.shape[0] == x.shape[1]:
        m = m.view(1, -1, 1, 1)
        s = s.view(1, -1, 1, 1)
    elif x.dim() == 3 and m.dim() == 1 and m.shape[0] == x.shape[0]:
        # single sample (C, H, W)
        m = m.view(-1, 1, 1)
        s = s.view(-1, 1, 1)
    elif x.dim() == 2 and m.dim() == 1 and m.shape[0] == x.shape[1]:
        # (B, D) — broadcast over batch
        m = m.view(1, -1)
        s = s.view(1, -1)
    return (x - m) / (s + 1e-8)


# ---------------------------------------------------------------------------
# Per-sensor preprocessing
# ---------------------------------------------------------------------------

def preprocess_daimon(raw, dataset_details, image_size, inference_transform='none'):
    """raw: (N, 384000) float32 from observation.tac0 → (N, 3, image_size, image_size).

    The raw daimon array is (240, 320, 5) per frame, flattened to 384000 in the
    HDF5. Select channels [0, 3, 4] = [depth, shear_x, shear_y] to match the
    training-time tensor built by preprocess.py::daimon_to_chw
    (the daimon_stats baked into the checkpoint correspond to *this* layout).
    """
    H, W, C = 240, 320, 5
    expected = H * W * C
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[-1] != expected:
        raise ValueError(
            f"daimon raw last-dim={raw.shape[-1]} != {expected} (=240*320*5); "
            "are you sure --sensor D is being fed observation.tac0 (raw 5-channel) "
            "and not observation.image.tac0 (tac*.mp4 visualization)?"
        )
    x = torch.from_numpy(raw).float().view(-1, H, W, C)        # (N, H, W, 5)
    x = torch.stack([x[..., 0], x[..., 3], x[..., 4]], dim=-1) # [depth, shear_x, shear_y]
    x = x.permute(0, 3, 1, 2).contiguous()                     # (N, 3, H, W)
    x = transforms.functional.resize(x, [image_size, image_size], antialias=True)
    stats = dataset_details.get('daimon_stats', None)
    if stats is not None:
        x = _zscore(x, stats['mean'], stats['std'])
    # inference_transform is a uniform flip across all 3 channels — fine to leave
    # as 'none' (the default). Mirror was a training-time cross-modal alignment;
    # for single-sensor policy use, encoder consistency between conversion-time
    # and deploy-time is what matters, not the absolute training orientation.
    x = _apply_transform(x, inference_transform)
    return x


def preprocess_eflesh(raw, dataset_details, inference_transform='none'):
    """raw: (N, 15) float32 → (N, 15) z-scored."""
    x = torch.from_numpy(raw).float()                          # (N, 15)
    stats = dataset_details.get('eflesh_stats', None)
    if stats is not None:
        x = _zscore(x, stats['mean'], stats['std'])
    # Vector transforms act on features, never on the temporal/batch axis.
    if inference_transform == 'fliplr':
        x = torch.flip(x, dims=[-1])
    return x


def preprocess_flexi(raw, dataset_details, inference_transform='none'):
    """raw: (N, 192) float32 → (N, 12, 16) z-scored."""
    x = torch.from_numpy(raw).float().reshape(-1, 12, 16)      # (N, 12, 16)
    stats = dataset_details.get('flexitac_stats', None)
    if stats is not None:
        # flexi stats are scalar; broadcast to channel-1
        m = float(stats['mean']) if not isinstance(stats['mean'], (list, tuple)) else stats['mean'][0]
        s = float(stats['std']) if not isinstance(stats['std'], (list, tuple)) else stats['std'][0]
        x = (x - m) / (s + 1e-8)
    x = _apply_transform(x, inference_transform)
    return x


# ---------------------------------------------------------------------------
# Encoder forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(model, x, sensor, device, latent_type='mu'):
    """x: (B, ...) on CPU. Returns (B, latent_dim) numpy."""
    x = x.to(device)
    h, mu, logvar, z = model._enc(x, sensor)
    if latent_type == 'mu':
        out = mu
    elif latent_type == 'z':
        out = z
    elif latent_type == 'h':
        out = h
    else:
        raise ValueError(f"Unknown latent_type: {latent_type}")
    return out.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Per-episode processing
# ---------------------------------------------------------------------------

_SENSOR_TO_HDF5 = {
    # Daimon: read RAW 5-channel array (shape [N, 240*320*5]) — NOT the tac*.mp4
    # video visualization at observation.image.tac0/tac1.
    'D': ('observation.tac0', 'observation.tac1'),
    'E': ('observation.tac0', 'observation.tac1'),
    'F': ('observation.tac0', 'observation.tac1'),
}


def process_episode(in_path, out_path, model, sensor, dataset_details, image_size,
                    batch_size, device, latent_type='mu', inference_transform='none',
                    overwrite=False):
    """Convert streams in bounded batches; preserve all unrelated HDF5 objects."""
    in_path, out_path = Path(in_path).resolve(), Path(out_path).resolve()
    if in_path == out_path:
        raise ValueError('Input and output episode paths must differ')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if out_path.exists() and not overwrite:
        raise FileExistsError(f'{out_path} exists; pass --overwrite to replace converted episodes')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix='.hdf5', delete=False) as f:
        temp_path = Path(f.name)
    try:
        with h5py.File(in_path, 'r') as fin, h5py.File(temp_path, 'w') as fout:
            keys = ('observation.tac0', 'observation.tac1')
            for key in keys:
                if key not in fin or len(fin[key]) == 0:
                    raise ValueError(f'{in_path}: missing or empty {key}')
            if len(fin[keys[0]]) != len(fin[keys[1]]):
                raise ValueError('Tactile streams have different frame counts')
            for key, value in fin.attrs.items():
                fout.attrs[key] = value
            for key in fin:
                if key not in keys:
                    fin.copy(key, fout)
            for key in keys:
                source = fin[key]
                dest = None
                for start in range(0, len(source), batch_size):
                    raw = source[start:start + batch_size]
                    if sensor == 'D':
                        x = preprocess_daimon(raw, dataset_details, image_size, inference_transform)
                    elif sensor == 'E':
                        if raw.ndim != 2 or raw.shape[1] != 15:
                            raise ValueError('eFlesh observations must have shape (T, 15)')
                        x = preprocess_eflesh(raw, dataset_details, inference_transform)
                    elif sensor == 'F':
                        if raw.ndim != 2 or raw.shape[1] != 192:
                            raise ValueError('FlexiTac observations must have shape (T, 192)')
                        x = preprocess_flexi(raw, dataset_details, inference_transform)
                    else:
                        raise ValueError(f'Unknown sensor: {sensor}')
                    latent = encode_batch(model, x, sensor, device, latent_type)
                    if not np.isfinite(latent).all():
                        raise FloatingPointError(f'Non-finite latents in {key}')
                    if dest is None:
                        dest = fout.create_dataset(key, shape=(len(source), latent.shape[1]), dtype='float32')
                        for attr, value in source.attrs.items():
                            dest.attrs[attr] = value
                        dest.attrs['representation'] = f'TactX/{latent_type}'
                        fout.attrs['tactile_dim'] = latent.shape[1]
                    dest[start:start + len(latent)] = latent
        os.replace(temp_path, out_path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True,
                        help='checkpoints/cross_modal_tri/cross_modal_vae3/dataset_new_run_...')
    parser.add_argument('--checkpoint_file', default='model.pt',
                        help='Weights file inside --checkpoint_dir (e.g. model_ep400.pt).')
    parser.add_argument('--sensor', required=True, choices=['D', 'E', 'F'],
                        help='Which sensor encoder to use.')
    parser.add_argument('--input_dir', required=True,
                        help='Directory of teleop HDF5 episodes.')
    parser.add_argument('--output_dir', required=True,
                        help='Destination for converted HDF5 episodes.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_type', choices=['mu', 'z', 'h'], default='mu')
    parser.add_argument('--inference_transform',
                        choices=['none', 'fliplr', 'flipud', 'rot180'],
                        default='none',
                        help='Optional geometric correction at inference. Apply '
                             'if teleop data orientation differs from the '
                             'training-time canonical (after configured per-pair rot180).')
    parser.add_argument('--stats_dataset_name', type=str, default=None,
                        help='If given, pull *_stats from this entry in '
                             'config/datasets_config.yaml instead of the '
                             'checkpoint-saved (static) dataset_details. Use '
                             'this when converting teleop / dynamic-regime data '
                             '(e.g. dataset_cross_modal_tri_dynamic_all).')
    args = parser.parse_args()

    if Path(args.input_dir).resolve() == Path(args.output_dir).resolve():
        parser.error('input_dir and output_dir must differ')
    if args.batch_size < 1:
        parser.error('batch_size must be positive')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    with open(os.path.join(args.checkpoint_dir, 'model_details.yaml')) as f:
        model_details = yaml.safe_load(f)
    with open(os.path.join(args.checkpoint_dir, 'dataset_details.yaml')) as f:
        dataset_details = yaml.safe_load(f)
    image_size = dataset_details.get('image_size', 224)

    # Optional override: replace *_stats and image_size with values from
    # another dataset entry (e.g. dynamic stats for teleop conversion).
    if args.stats_dataset_name:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_root, 'config', 'datasets_config.yaml')) as f:
            cfg = yaml.safe_load(f)
        if args.stats_dataset_name not in cfg:
            raise KeyError(f"{args.stats_dataset_name!r} not in datasets_config.yaml")
        override = cfg[args.stats_dataset_name]
        for k in ('daimon_stats', 'eflesh_stats', 'flexitac_stats', 'image_size'):
            if k in override:
                dataset_details[k] = override[k]
        image_size = dataset_details.get('image_size', 224)
        print(f"[stats override] using {args.stats_dataset_name} for *_stats")

    model = get_model('cross_modal_vae3', model_details).to(args.device)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_file),
                      map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model.eval()

    print('=' * 60)
    print(f"Sensor: {args.sensor}   latent_type: {args.latent_type}   "
          f"inference_transform: {args.inference_transform}")
    print(f"Input : {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print('=' * 60)

    eps = sorted(glob.glob(os.path.join(args.input_dir, 'episode_*.hdf5')))
    if not eps:
        raise FileNotFoundError(f"No episode_*.hdf5 in {args.input_dir}")

    for ep_path in tqdm(eps, desc=f"convert[{args.sensor}]"):
        out_path = os.path.join(args.output_dir, os.path.basename(ep_path))
        process_episode(ep_path, out_path, model, args.sensor,
                        dataset_details, image_size, args.batch_size,
                        args.device, args.latent_type, args.inference_transform, args.overwrite)
        tqdm.write(f"  → {out_path}")

    print(f"\nDone. {len(eps)} episodes written to {args.output_dir}/")


if __name__ == '__main__':
    main()
