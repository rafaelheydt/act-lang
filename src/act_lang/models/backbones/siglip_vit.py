"""Backbone visual: torre de imagem do SigLIP 2 (Tschannen et al., 2025),
base-patch16-224.

Ver docstring de clip_vit.py -- mesmo raciocínio e mesmo contrato de
VisionBackbone. Diferenças específicas do SigLIP em relação ao CLIP,
confirmadas em scripts/diagnose_siglip_patch_alignment.py e reaproveitadas
aqui literalmente:
  - SigLIP NÃO usa token [CLS] -- last_hidden_state já é só a grade de
    patches, sem descartar nada.
  - Normalização própria do checkpoint é [0.5, 0.5, 0.5] (mean e std), não
    as estatísticas do ImageNet -- lida via AutoImageProcessor, nunca
    hardcoded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"


class SiglipVisionBackbone(nn.Module):
    def __init__(self, d_model: int, pretrained: bool = True, model_name: str = DEFAULT_SIGLIP_MODEL):
        super().__init__()
        from transformers import AutoImageProcessor, SiglipVisionConfig, SiglipVisionModel

        if pretrained:
            self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        else:
            self.vision_model = SiglipVisionModel(SiglipVisionConfig.from_pretrained(model_name))

        config = self.vision_model.config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.grid = self.image_size // self.patch_size
        self.proj = nn.Conv2d(config.hidden_size, d_model, kernel_size=1)

        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) em [0, 1] -> feature map (B, d_model, grid, grid)."""
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        images = (images - self.img_mean) / self.img_std
        hidden = self.vision_model(pixel_values=images).last_hidden_state  # (B, n_patches, hidden) -- sem CLS
        b, n, c = hidden.shape
        feat_map = hidden.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return self.proj(feat_map)
