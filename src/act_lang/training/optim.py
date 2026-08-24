"""Optimizer com grupos de parâmetros.

CORREÇÃO aplicada: o ACT original usa lr_backbone 10x menor que o resto
(herança do DETR) — fine-tuning suave da ResNet pré-treinada em vez de
sobrescrever o pré-treino no início do treinamento.
"""

import torch


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    backbone_prefix = "vision_backbone.backbone"
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if name.startswith(backbone_prefix) else other_params).append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )
