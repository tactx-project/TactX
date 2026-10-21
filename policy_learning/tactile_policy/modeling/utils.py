"""Image conversion used by the ACT demonstration loader."""
import torch


def unit_normalization(image_nchw):
    """Convert uint8 camera frames (N, C, H, W) to float tensors in [0, 1]."""
    return torch.from_numpy(image_nchw).float() / 255
