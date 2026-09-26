"""Optimizer com grupos de parâmetros.

CORREÇÃO aplicada: o ACT original usa lr_backbone 10x menor que o resto
(herança do DETR) — fine-tuning suave da ResNet pré-treinada em vez de
sobrescrever o pré-treino no início do treinamento.

CORREÇÃO (backbones pareados CLIP/SigLIP): o prefixo usado pra separar o
grupo de LR do backbone era "vision_backbone.backbone" -- batia com o
atributo interno `.backbone` do VisionBackbone (ResNet18), mas não com
`.vision_model` do CLIPVisionBackbone/SiglipVisionBackbone. Generalizado
para "vision_backbone." (todo o backbone visual, não só o sub-atributo
pré-treinado) -- também corrige uma inconsistência pré-existente: o `proj`
(Conv1x1, treinado do zero em qualquer um dos backbones) passa a cair em
lr_backbone junto com o resto do backbone visual, em vez de lr normal.
"""

import torch


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    backbone_prefix = "vision_backbone."
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
