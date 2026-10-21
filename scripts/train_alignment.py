"""Train TactX alignment or a documented objective baseline on paired contacts."""
import argparse
from datetime import datetime
from pathlib import Path

import wandb
import yaml

from joint_embedding_learning.data.datasets import get_cross_modal_tri_datasets
from joint_embedding_learning.training import train_tri

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJECTS = ['circle', 'cylinder', 'cylinder_vertical', 'ellipse', 'large_ball',
                   'plane', 'plane_horizontal', 'small_ball', 'square', 'triangle']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_name', default='dataset_new')
    parser.add_argument('--dataset_config', type=Path, default=REPO_ROOT / 'config/datasets_config.yaml')
    parser.add_argument('--model_config', type=Path, default=REPO_ROOT / 'config/models_config.yaml')
    parser.add_argument('--model_name', choices=['cross_modal_vae3', 'cross_modal_clr3'], default='cross_modal_vae3')
    parser.add_argument('--data_root', type=Path, help='Override the paired-data directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run_name', default='tactx')
    parser.add_argument('--objects', nargs='+', default=DEFAULT_OBJECTS)
    parser.add_argument('--resume_from', help='Warm-start weights; does not resume optimizer or epoch state')
    parser.add_argument('--alignment_loss', choices=['infonce', 'l2'], default='infonce')
    for name in ['projection_dim', 'epochs', 'batch_size', 'infonce_warmup_start', 'infonce_warmup_epochs']:
        parser.add_argument(f'--{name}', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help='Enable CUDA mixed precision')
    parser.add_argument('--checkpoint_root', type=Path, default=REPO_ROOT / 'checkpoints')
    for pair in ['de', 'ef', 'fd']:
        parser.add_argument(f'--{pair}_stride', type=int, default=1)
    args = parser.parse_args()
    if args.model_name == 'cross_modal_clr3' and args.alignment_loss != 'infonce':
        parser.error('The contrastive-only model requires --alignment_loss infonce')
    if args.num_workers < 0 or any(getattr(args, f'{p}_stride') < 1 for p in ['de', 'ef', 'fd']):
        parser.error('num_workers must be nonnegative and frame strides must be positive')
    with args.dataset_config.open() as f:
        datasets = yaml.safe_load(f)
    if args.dataset_name not in datasets:
        parser.error(f'Unknown dataset {args.dataset_name}; available: {list(datasets)}')
    dataset_details = dict(datasets[args.dataset_name])
    data_root = args.data_root or Path(dataset_details['data_root'])
    dataset_details['data_root'] = str(data_root.resolve() if args.data_root else (REPO_ROOT / data_root).resolve())
    dataset_details['objects'] = args.objects
    pair_strides = {p: getattr(args, f'{p}_stride') for p in ['de', 'ef', 'fd']}
    dataset_details['pair_strides'] = pair_strides
    with args.model_config.open() as f:
        model_details = dict(yaml.safe_load(f)[args.model_name])
    for name in ['projection_dim', 'epochs', 'batch_size', 'infonce_warmup_start', 'infonce_warmup_epochs']:
        if getattr(args, name) is not None:
            model_details[name] = getattr(args, name)
    if args.lr is not None:
        model_details['learning_rate'] = args.lr
    model_details.update(loss=args.alignment_loss, num_workers=args.num_workers, amp=args.amp,
                         checkpoint_root=str(args.checkpoint_root.resolve()), model_name=args.model_name)
    train, val, _ = get_cross_modal_tri_datasets(args.dataset_name, dataset_details,
                                                objects=args.objects, pair_strides=pair_strides)
    run_name = f'{args.run_name}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    with wandb.init(project='TactX', name=run_name, config={'model': model_details, 'dataset': dataset_details}):
        path = train_tri(train, {'val': val}, args.dataset_name, dataset_details,
                         args.model_name, model_details, wandb, run_name=run_name,
                         device=args.device, resume_from=args.resume_from)
    print(f'Checkpoint directory: {path}')


if __name__ == '__main__':
    main()
