"""Offline ACT training with frozen TactX tactile representations.

The policy and demonstration utilities retain adaptations from the Wang Lab
codebase and ACT/DETR; see ../../THIRD_PARTY_NOTICES.md.
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, send_to_device
from omegaconf import OmegaConf
from tqdm import tqdm

from tactile_policy.data_utils_hdt import load_data, set_seed
from tactile_policy.modeling.modeling_act import ACTPolicy
from tactile_policy.modeling.utils import unit_normalization

PACKAGE_ROOT = Path(__file__).resolve().parent
POLICY_ROOT = PACKAGE_ROOT.parent


def config_path(value, folder, suffix):
    path = Path(value)
    return path if path.is_file() else PACKAGE_ROOT / 'configs' / folder / f'{value}{suffix}'


def make_policy_config(model_path, lr, chunk_size=None, pretrained_backbone=True):
    trainer = OmegaConf.to_container(OmegaConf.load(model_path), resolve=True)
    common = trainer['common']
    if common['policy_class'] != 'ACT':
        raise ValueError('This package contains the ACT policy workflow')
    sensor_path = Path(model_path).parent.parent / 'sensors' / f"{common['sensor_config']}.yaml"
    sensor = OmegaConf.to_container(OmegaConf.load(sensor_path), resolve=True)
    config = dict(trainer['model'])
    config.update({k: common[k] for k in ['state_dim', 'action_dim', 'camera_names', 'tactile_names']})
    config.update({k: v for k, v in sensor.items() if k.startswith('tactile_') or k == 'lr_tactile_backbone'})
    config['tactile_modality'] = sensor['modality']
    config['lr'] = lr
    config['lr_backbone'] = float(config['lr_backbone'])
    config['lr_tactile_backbone'] = float(config['lr_tactile_backbone'])
    config['chunk_size'] = chunk_size or int(config['chunk_size'])
    config['num_queries'] = config['chunk_size']
    config['pretrained_backbone'] = pretrained_backbone
    return config


@torch.no_grad()
def validate(policy, loader, accelerator, max_batches):
    policy.eval()
    total, count = 0.0, 0
    for data in loader:
        output = policy(*send_to_device(data, accelerator.device))
        loss = output['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite ACT validation loss')
        total += loss.item()
        count += 1
        if count >= max_batches:
            break
    policy.train()
    if count == 0:
        raise ValueError('Validation loader is empty')
    return total / count


def train(args):
    set_seed(args.seed)
    model_path = config_path(args.model_cfg, 'models', '.yaml')
    data_path = config_path(args.data_cfg, '', '.yaml')
    split_path = config_path(args.dataset_json, 'datasets', '.json')
    policy_config = make_policy_config(model_path, args.lr, args.chunk_size, not args.no_pretrained_backbone)
    data_config = OmegaConf.to_container(OmegaConf.load(data_path), resolve=True)['data']
    ckpt_dir = Path(args.checkpoint_root) / f'{args.exptid}_ckpt'
    accelerator = Accelerator(cpu=args.cpu, gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    policy = ACTPolicy(policy_config)
    optimizer = policy.configure_optimizers()
    train_loader, val_loader, stats = load_data(
        str(args.data_root), data_config, str(split_path), policy_config['camera_names'],
        policy_config['chunk_size'], args.batch_size, args.batch_size, unit_normalization,
        tactile_names=policy_config['tactile_names'], tactile_modality=policy_config['tactile_modality'],
        num_workers=args.num_workers)
    update_steps = math.ceil(args.num_steps / args.gradient_accumulation_steps)
    if update_steps < 3:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=update_steps)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=update_steps,
            pct_start=max(0.1, 2.0 / update_steps), anneal_strategy='cos')
    # Step the scheduler explicitly only on optimizer updates, including accumulation.
    policy, optimizer, train_loader = accelerator.prepare(policy, optimizer, train_loader)
    accelerator.register_for_checkpointing(scheduler)
    if args.load_pretrained_path:
        state = torch.load(args.load_pretrained_path, map_location='cpu', weights_only=True)
        accelerator.unwrap_model(policy).load_state_dict(state, strict=True)
    set_seed(args.seed + accelerator.process_index * 1000)
    step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        with (Path(args.resume_from) / 'training_state.json').open() as f:
            step = json.load(f)['step']
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / 'dataset_stats.pkl').open('wb') as f:
            pickle.dump(stats, f)
        (ckpt_dir / 'policy_config.json').write_text(json.dumps(policy_config, indent=2))
        wandb.init(project='TactX-policy', name=args.exptid,
                   mode='disabled' if args.no_wandb else 'online', config=policy_config)
    accelerator.wait_for_everyone()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    for step in tqdm(range(step, args.num_steps), desc='ACT steps', disable=not accelerator.is_main_process):
        if step % args.val_every == 0:
            val_loss = validate(policy, val_loader, accelerator, args.val_batches)
            if accelerator.is_main_process:
                wandb.log({'val/loss': val_loss}, step=step)
                print(f'Step {step}: validation loss {val_loss:.6f}')
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            data = next(iterator)
        with accelerator.accumulate(policy):
            output = policy(*data)
            loss = output['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite ACT loss at step {step}')
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if accelerator.is_main_process:
            wandb.log({f'train/{k}': v.detach().item() for k, v in output.items()}, step=step + 1)
        if (step + 1) % args.save_every == 0:
            state_dir = ckpt_dir / f'policy_iter_{step + 1}_seed_{args.seed}'
            accelerator.save_state(str(state_dir), safe_serialization=False)
            if accelerator.is_main_process:
                (state_dir / 'training_state.json').write_text(json.dumps({'step': step + 1}))
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(policy)
    accelerator.save(state, ckpt_dir / 'policy_last.ckpt')
    if accelerator.is_main_process:
        wandb.finish()
        print(f'Saved model to {ckpt_dir / "policy_last.ckpt"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_steps', '--num_epochs', dest='num_steps', type=int, required=True,
                        help='Number of training batches; num_epochs is retained as an alias')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--exptid', required=True)
    parser.add_argument('--dataset_json', required=True, help='Included config name or JSON path')
    parser.add_argument('--model_cfg', default='act_cross_modal_pnp_z16', help='Included config name or YAML path')
    parser.add_argument('--data_cfg', default='data_base')
    parser.add_argument('--data_root', type=Path, default=POLICY_ROOT / 'data/recordings/processed')
    parser.add_argument('--checkpoint_root', type=Path, default=POLICY_ROOT / 'data/checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=1000)
    parser.add_argument('--val_batches', type=int, default=21)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--load_pretrained_path', help='Warm-start an ACT state dictionary')
    parser.add_argument('--resume_from', help='Resume an Accelerate checkpoint directory with identical training settings')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_pretrained_backbone', action='store_true', help='Use random visual weights; intended for offline smoke tests')
    args = parser.parse_args()
    if any(getattr(args, k) <= 0 for k in ['batch_size', 'num_steps', 'gradient_accumulation_steps', 'val_every', 'val_batches', 'save_every']) or args.num_workers < 0:
        parser.error('Counts must be positive and num_workers nonnegative')
    if args.num_steps % args.gradient_accumulation_steps:
        parser.error('num_steps must be divisible by gradient_accumulation_steps')
    train(args)


if __name__ == '__main__':
    main()
