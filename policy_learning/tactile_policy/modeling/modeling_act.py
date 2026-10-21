import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2
import torch
from huggingface_hub import PyTorchModelHubMixin

from tactile_policy.third_party.detr.main import build_ACT_model_and_optimizer

class ACTPolicy(nn.Module, PyTorchModelHubMixin):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.tactile_modality = args_override['tactile_modality']
        # patch_h = 24
        # patch_w = 42
        patch_h = 16
        patch_w = 22
        # FIXME(roger): RSS temp. EEF loss
        self.USE_EEF_LOSS = True
        self.transform = v2.Compose([
            v2.Resize((patch_h * 14, patch_w * 14)),
            # v2.CenterCrop((patch_h * 14, patch_w * 14)),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        print(f'KL Weight {self.kl_weight}')
        print(f'Tactile Modality: {self.tactile_modality}')

    def forward(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None, tactile_images=None, tactile_arrays=None):
        env_state = None

        image = self.transform(image)

        # Prepare tactile data based on modality
        tactile_data = None
        if self.tactile_modality == 'image' and tactile_images is not None and tactile_images.numel() > 0:
            tactile_data = self.transform(tactile_images)
        elif self.tactile_modality in ['array', 'state'] and tactile_arrays is not None and tactile_arrays.numel() > 0:
            tactile_data = tactile_arrays

        if actions is not None:
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, tactile_data, actions, is_pad, conditioning_dict)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = (all_l1 * ~is_pad.unsqueeze(-1))

            loss_dict['l1'] = all_l1.mean()
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else:
            a_hat, _, (_, _) = self.model(qpos, image, env_state, tactile_data, conditioning_dict=conditioning_dict)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
