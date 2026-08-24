"""Perda do ACT: L1 mascarada + KL com free bits (validado no Push-T)."""

import torch
import torch.nn.functional as F


def masked_l1(
    pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor
) -> torch.Tensor:
    """L1 média sobre os elementos VÁLIDOS (padding excluído do denominador)."""
    abs_err = F.l1_loss(pred, target, reduction="none")
    valid_mask = (~is_pad).unsqueeze(-1)
    num_valid = (valid_mask.sum() * abs_err.shape[-1]).clamp_min(1)
    return (abs_err * valid_mask).sum() / num_valid


def kld_free_bits(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.05
) -> torch.Tensor:
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_clamped = torch.clamp(kld_per_dim - free_bits, min=0.0)
    return kld_clamped.sum(dim=1).mean(dim=0)


def act_loss(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    is_pad: torch.Tensor,
    kl_weight: float = 10.0,
    free_bits: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retorna (loss total, recon, kld_raw).

    O gradiente usa o KL penalizado (free bits); o kld_raw é retornado apenas
    para logging — monitorar o KL real enquanto se otimiza o clampado.
    """
    recon_loss = masked_l1(pred_actions, gt_actions, is_pad)

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_raw = klds.sum(dim=1).mean(dim=0)
    kld_penalizado = kld_free_bits(mu, logvar, free_bits)

    return recon_loss + kl_weight * kld_penalizado, recon_loss, kld_raw
