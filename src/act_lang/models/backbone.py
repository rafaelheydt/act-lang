"""Shim de compatibilidade -- o conteúdo real mudou para models/backbones/
(pacote com um backbone por arquivo: resnet.py, clip_vit.py, siglip_vit.py,
mais factory.py). Mantido para não quebrar imports existentes
(`from act_lang.models.backbone import VisionBackbone, freeze_batchnorm`).
"""

from .backbones.resnet import VisionBackbone, freeze_batchnorm

__all__ = ["VisionBackbone", "freeze_batchnorm"]
