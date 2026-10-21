"""TactX and the documented reconstruction, contrastive, and L2 baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from simclr.modules import NT_Xent

def kl_divergence_standard(mu, logvar):
    """KL[ q(z|x) || N(0, I) ] averaged over batch and latent dims.

    mu, logvar: (B, D)
    Returns: scalar tensor.
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()

class CrossModalVAE3Loss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 (joint pair-only training).

    Forward inputs:
        outs: dict with keys 'de', 'ef', 'fd', each value is the per-pair dict
              from CrossModalVAE3.forward_pair (contains mu_a, logvar_a, z_a,
              mu_b, logvar_b, z_b, recon_aa, recon_ba, recon_bb, recon_ab,
              sensor_a, sensor_b).
        targets: dict with keys 'de', 'ef', 'fd', each value is (x_a, x_b)
                 raw input tensors used as reconstruction targets.
        beta: float, KL weight applied this step (warm-up scheduler externally).

    Loss = sum_{p in {de,ef,fd}} L_InfoNCE(mu_a^p, mu_b^p)
         + lambda_recon * sum_{p} (L1(recon_aa, x_a) + L1(recon_ba, x_a)
                                   + L1(recon_bb, x_b) + L1(recon_ab, x_b))
         + beta * sum_{p} (KL(mu_a^p, logvar_a^p) + KL(mu_b^p, logvar_b^p))

    The same N(0, I) prior anchors all three modalities. Pair-wise InfoNCE
    directly aligns all three observed pairs DE / EF / FD.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size, temperature, lambda_recon=1.0):
        super().__init__()
        if NT_Xent is None:
            raise ImportError("simclr.modules.NT_Xent is required for CrossModalVAE3Loss")
        # Three independent NT_Xent objects (state-free, but keep symmetric API).
        self.nt_xent_de = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_ef = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_fd = NT_Xent(batch_size, temperature, 1)
        self.lambda_recon = float(lambda_recon)

    def _nt_for(self, pair, batch_size):
        criterion = getattr(self, f'nt_xent_{pair}')
        if criterion.batch_size != batch_size:
            criterion = NT_Xent(batch_size, criterion.temperature, 1)
            setattr(self, f'nt_xent_{pair}', criterion)
        return criterion

    @staticmethod
    def _shape_match(recon, target):
        """Reshape recon to match target shape if a singleton channel needs squeezing."""
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            alpha: InfoNCE weight (0→1 curriculum). When 0 the model trains
                   reconstruction-only so the latent space captures contact
                   information before contrastive alignment is introduced.
            meta: accepted and ignored — kept so all tri losses share one
                  call signature.
        """
        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}  # keyed by f"{pair}_{a|b}"

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]

            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            L_info = self._nt_for(pair, len(mu_a))(mu_a, mu_b)
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info

class CrossModalVAE3UniforceLoss(nn.Module):
    """Tri-modal VAE loss for CrossModalVAE3 — 'uniforce' variant.

    Identical to CrossModalVAE3Loss in every respect EXCEPT the cross-modal
    alignment term: instead of pair-wise InfoNCE (contrastive, with negatives)
    it uses a plain L2 (MSE) pull between the matched paired means:

        L_align = sum_{p in {de,ef,fd}}  MSE(mu_a^p, mu_b^p)

    Recon (4× L1 per pair) and KL are unchanged.

        Loss = alpha * L_align
             + lambda_recon * sum_p (L1 recon, 4 branches)
             + beta        * sum_p (KL_a + KL_b)

    Output dict keys are kept identical to CrossModalVAE3Loss (``infonce``,
    ``infonce_{pair}``, ``alpha_infonce``, ...) so the existing tri training /
    validation / logging code consumes it unchanged — for this loss those
    keys carry the L2 alignment value, not an InfoNCE value.

    Note on collapse: contrastive alignment resists collapse via negatives;
    a pure L2 pull does not. Here the *only* anti-collapse force is the
    reconstruction term — different contacts must decode to different inputs,
    so the latent cannot collapse to a point. Keep ``lambda_recon`` healthy
    relative to ``alpha`` or the means will collapse toward 0.

    __init__ signature matches CrossModalVAE3Loss (batch_size / temperature
    accepted and ignored) so get_loss can construct it the same way.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size=None, temperature=None, lambda_recon=1.0):
        super().__init__()
        self.lambda_recon = float(lambda_recon)

    @staticmethod
    def _shape_match(recon, target):
        """Reshape recon to match target shape if a singleton channel needs squeezing."""
        if recon.shape == target.shape:
            return recon
        if recon.dim() == target.dim() + 1 and recon.shape[1] == 1:
            return recon.squeeze(1)
        if target.dim() == recon.dim() + 1 and target.shape[1] == 1:
            return recon.unsqueeze(1)
        return recon

    def forward(self, outs, targets, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            alpha: alignment weight (0→1 curriculum). When 0 the model trains
                   reconstruction-only, identical to the InfoNCE variant.
            meta: accepted and ignored — kept so all tri losses share one
                  call signature.
        """
        info_per_pair = {}
        recon_per_pair = {}
        kl_per_sensor_pair = {}  # keyed by f"{pair}_{a|b}"

        L_info_total = 0.0
        L_recon_total = 0.0
        L_kl_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            x_a, x_b = targets[pair]

            mu_a, mu_b = out["mu_a"], out["mu_b"]
            logvar_a, logvar_b = out["logvar_a"], out["logvar_b"]

            # --- uniforce: L2 alignment instead of InfoNCE ---
            L_info = F.mse_loss(mu_a, mu_b)
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

            r_aa = F.l1_loss(self._shape_match(out["recon_aa"], x_a), x_a)
            r_ba = F.l1_loss(self._shape_match(out["recon_ba"], x_a), x_a)
            r_bb = F.l1_loss(self._shape_match(out["recon_bb"], x_b), x_b)
            r_ab = F.l1_loss(self._shape_match(out["recon_ab"], x_b), x_b)
            L_recon_pair = r_aa + r_ba + r_bb + r_ab
            L_recon_total = L_recon_total + L_recon_pair
            recon_per_pair[pair] = {
                "total": L_recon_pair.detach(),
                "recon_aa": r_aa.detach(), "recon_ba": r_ba.detach(),
                "recon_bb": r_bb.detach(), "recon_ab": r_ab.detach(),
            }

            kl_a = kl_divergence_standard(mu_a, logvar_a)
            kl_b = kl_divergence_standard(mu_b, logvar_b)
            L_kl_total = L_kl_total + kl_a + kl_b
            kl_per_sensor_pair[f"{pair}_a"] = kl_a.detach()
            kl_per_sensor_pair[f"{pair}_b"] = kl_b.detach()

        total = alpha * L_info_total + self.lambda_recon * L_recon_total + beta * L_kl_total

        info = {
            "total": total,
            "infonce": L_info_total.detach(),          # carries L2 align value
            "recon": L_recon_total.detach(),
            "kl": L_kl_total.detach() if torch.is_tensor(L_kl_total) else torch.tensor(float(L_kl_total)),
            "beta": torch.tensor(float(beta)),
            "alpha_infonce": torch.tensor(float(alpha)),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = recon_per_pair[pair]["total"]
            for k in ("recon_aa", "recon_ba", "recon_bb", "recon_ab"):
                info[f"{k}_{pair}"] = recon_per_pair[pair][k]
        for k, v in kl_per_sensor_pair.items():
            info[f"kl_{k}"] = v
        return info

class CrossModalCLR3Loss(nn.Module):
    """Tri-modal *plain contrastive* loss for CrossModalCLR3.

    Pair-wise NT-Xent over the de / ef / fd projections — and nothing else.
    No reconstruction, no KL: this is the lean contrastive baseline.

        Loss = sum_{p in {de,ef,fd}}  NT-Xent(z_a^p, z_b^p)

    Contrastive alignment is applied at FULL weight from epoch 0. The ``alpha``
    (InfoNCE curriculum) and ``beta`` (KL) arguments are accepted only for
    call-signature compatibility with the VAE3 losses and are ignored — there
    is no recon term to warm up before, and no KL term.

    The returned info dict keeps the CrossModalVAE3Loss keys (``infonce``,
    ``recon``, ``kl``, ``infonce_{pair}``, ``recon_{pair}``, ``alpha_infonce``,
    ``beta``) so the existing tri training / validation / logging code consumes
    it unchanged — ``recon`` and ``kl`` (and their per-pair entries) are
    reported as a constant 0.
    """

    PAIR_NAMES = ("de", "ef", "fd")

    def __init__(self, batch_size, temperature, lambda_recon=None):
        super().__init__()
        if NT_Xent is None:
            raise ImportError("simclr.modules.NT_Xent is required for CrossModalCLR3Loss")
        # Three independent NT_Xent objects (state-free, but keep symmetric API).
        self.nt_xent_de = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_ef = NT_Xent(batch_size, temperature, 1)
        self.nt_xent_fd = NT_Xent(batch_size, temperature, 1)

    def _nt_for(self, pair, batch_size):
        criterion = getattr(self, f'nt_xent_{pair}')
        if criterion.batch_size != batch_size:
            criterion = NT_Xent(batch_size, criterion.temperature, 1)
            setattr(self, f'nt_xent_{pair}', criterion)
        return criterion

    def forward(self, outs, targets=None, beta=0.0, alpha=1.0, meta=None):
        """
        Args:
            outs: dict {pair: per-pair dict from CrossModalCLR3.forward_pair}.
            targets, beta, alpha, meta: accepted and ignored (signature parity
                with the VAE3 losses — no recon, no KL, no curriculum here).
        """
        info_per_pair = {}
        L_info_total = 0.0

        for pair in self.PAIR_NAMES:
            out = outs[pair]
            L_info = self._nt_for(pair, len(out["z_a"]))(out["z_a"], out["z_b"])
            L_info_total = L_info_total + L_info
            info_per_pair[pair] = L_info.detach()

        total = L_info_total
        zero = torch.zeros((), device=total.device)

        info = {
            "total": total,
            "infonce": L_info_total.detach(),
            "recon": zero,
            "kl": zero,
            "beta": torch.tensor(0.0),
            "alpha_infonce": torch.tensor(1.0),
        }
        for pair in self.PAIR_NAMES:
            info[f"infonce_{pair}"] = info_per_pair[pair]
            info[f"recon_{pair}"] = zero
        return info
