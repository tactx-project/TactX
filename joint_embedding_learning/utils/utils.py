"""Optimization, objective selection, and checkpoint serialization."""
import os
import torch
import yaml
from joint_embedding_learning.utils.losses import CrossModalVAE3Loss, CrossModalVAE3UniforceLoss, CrossModalCLR3Loss


def load_optimizer(args, model):
    if args.optimizer != 'Adam':
        raise ValueError("The released alignment models support optimizer='Adam'")
    return torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                            weight_decay=getattr(args, 'weight_decay', 0.0)), None, None


def get_loss(args, model_name):
    loss_type = getattr(args, 'loss', 'infonce')
    if model_name == 'cross_modal_clr3':
        if loss_type != 'infonce':
            raise ValueError('cross_modal_clr3 supports only infonce')
        return CrossModalCLR3Loss(args.batch_size, args.temperature)
    if model_name != 'cross_modal_vae3' or loss_type not in ('infonce', 'l2', 'uniforce'):
        raise ValueError(f'Unsupported model/objective: {model_name}/{loss_type}')
    cls = CrossModalVAE3Loss if loss_type == 'infonce' else CrossModalVAE3UniforceLoss
    return cls(args.batch_size, args.temperature, getattr(args, 'lambda_recon', 1.0))


def save_model(model, dataset_name, dataset_type, model_type, dataset_details, model_details, run_name="", tag=None, checkpoint_root=None):
    root = checkpoint_root or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints')
    path_out = os.path.join(root, dataset_type, model_type, dataset_name + '_run_' + run_name)
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    torch.save(model.state_dict(), os.path.join(path_out, "model.pt"))

    # Periodic weight snapshot (kept, not overwritten) — e.g. tag='ep150'.
    # Only the weights; the yamls below are identical across epochs.
    if tag is not None:
        torch.save(model.state_dict(), os.path.join(path_out, f"model_{tag}.pt"))

    # Save dataset details as yaml file
    dataset_details_path = os.path.join(path_out, "dataset_details.yaml")
    with open(dataset_details_path, 'w') as file:
        yaml.dump(dataset_details, file)

    # Save model details as yaml file
    model_details_path = os.path.join(path_out, "model_details.yaml")
    with open(model_details_path, 'w') as file:
        yaml.dump(model_details, file)
    return path_out
