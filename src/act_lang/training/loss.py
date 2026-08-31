"""Perda do ACT: L1 mascarada + KL (fiel ao oficial por padrão, free bits opcional)."""

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
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """free_bits=0.0 (padrão) equivale ao KL cru do ACT oficial: como o KL
    entre duas Gaussianas nunca é negativo, `clamp(kld - 0, min=0) = kld`
    sem alterar nada. Passe free_bits > 0 para reativar a técnica de free
    bits (Kingma et al.), validada à parte no Push-T — não faz parte do
    ACT original."""
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_clamped = torch.clamp(kld_per_dim - free_bits, min=0.0)
    return kld_clamped.sum(dim=1).mean(dim=0)


def kl_weight_schedule(epoch: int, target_kl_weight: float, warmup_epochs: int) -> float:
    """kl_weight sobe linearmente de 0 até target_kl_weight ao longo de
    `warmup_epochs`, depois permanece fixo (annealing, Bowman et al. 2016,
    "Generating Sentences from a Continuous Space"). Motivação: no começo do
    treino, com kl_weight baixo, o decoder não paga quase nada por usar z --
    tende a criar o hábito de depender dele pra reconstruir melhor. Só depois
    a pressão pra comprimir z sobe, mas o decoder já aprendeu a usá-lo.

    warmup_epochs=0 (padrão) desliga o annealing -- kl_weight fixo em
    target_kl_weight desde a primeira época, comportamento idêntico a antes
    desta mudança.
    """
    if warmup_epochs <= 0:
        return target_kl_weight
    return target_kl_weight * min(1.0, epoch / warmup_epochs)


def act_loss(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    is_pad: torch.Tensor,
    kl_weight: float = 10.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retorna (loss total, recon, kld_raw).

    Com free_bits=0.0 (padrão), o gradiente usa o KL cru — mesmo
    comportamento do ACT oficial (kl_divergence em policy.py, sem clamp).
    kld_raw é sempre o KL sem clamp, útil pra logging mesmo se free_bits > 0.
    """
    recon_loss = masked_l1(pred_actions, gt_actions, is_pad)

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_raw = klds.sum(dim=1).mean(dim=0)
    kld_penalizado = kld_free_bits(mu, logvar, free_bits)

    return recon_loss + kl_weight * kld_penalizado, recon_loss, kld_raw