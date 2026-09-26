"""Backbone visual: torre de imagem do CLIP (Radford et al., 2021), ViT-B/32.

Ablação (ver docs/hdf5_migration.md e o plano de "backbones pareados"): troca
o ResNet18 pré-treinado por classificação ImageNet por um ViT pré-treinado
com contraste imagem-texto -- a aposta é que features contrastivas sejam
mais ricas/transferíveis, mesmo sem usar o alinhamento texto-imagem do CLIP
como sinal explícito (isso já foi testado à parte em
scripts/diagnose_siglip_patch_alignment.py e achado fraco/ruidoso pras
imagens de câmera de robô do LIBERO -- fora da distribuição desses modelos).

Mesmo contrato de VisionBackbone (resnet.py): `forward(images (B,3,H,W) em
[0,1]) -> (B, d_model, h, w)`. Diferenças internas:
  - ViT produz uma SEQUÊNCIA de patches (B, n_patches[+1 CLS], hidden_size),
    não um mapa conv -- reshape de volta pra (B, hidden_size, grid, grid)
    pra preservar o contrato espacial que o resto do ACT espera
    (encode_observations faz flatten(2).transpose(1,2) por cima disso).
  - CLIP prepende um token [CLS] em last_hidden_state (posição 0) -- é
    descartado aqui; SigLIP (siglip_vit.py) não tem CLS.
  - Normalização e resolução são as do PROCESSOR do checkpoint (via
    AutoImageProcessor), não as do ImageNet -- CLIP ViT-B/32 espera
    224x224; um resize (F.interpolate) dentro do forward trata tanto o
    pipeline HDF5 (128x128) quanto o JPEG (256x256) sem duplicar dado.
  - freeze_batchnorm (resnet.py) é um no-op seguro aqui -- CLIP usa
    nn.LayerNorm, não nn.BatchNorm2d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPVisionBackbone(nn.Module):
    def __init__(self, d_model: int, pretrained: bool = True, model_name: str = DEFAULT_CLIP_MODEL):
        super().__init__()
        from transformers import AutoImageProcessor, CLIPVisionConfig, CLIPVisionModel

        if pretrained:
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        else:
            self.vision_model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(model_name))

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
        hidden = self.vision_model(pixel_values=images).last_hidden_state  # (B, n_patches+1, hidden) -- CLS na posição 0
        hidden = hidden[:, 1:, :]  # descarta o CLS -- só os patches formam a grade espacial
        b, n, c = hidden.shape
        feat_map = hidden.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return self.proj(feat_map)
