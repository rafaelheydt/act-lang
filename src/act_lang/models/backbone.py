"""
Backbone visual (ResNet18) compartilhado entre câmeras.
"""

import torch
import torch.nn as nn
import torchvision


class VisionBackbone(nn.Module):
    def __init__(self, d_model: int, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        # Estatísticas que os pesos IMAGENET1K_V1 esperam na entrada.
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) em [0, 1] -> feature map (B, d_model, h, w)."""
        images = (images - self.img_mean) / self.img_std
        features = self.backbone(images)
        return self.proj(features)


def freeze_batchnorm(module: nn.Module) -> nn.Module:
    """Substitui todo BatchNorm2d por FrozenBatchNorm2d (padrão DETR/ACT original).

    Opcional, mas recomendado: BN com batches pouco diversos de imagens de robô
    degrada as estatísticas do pré-treino e cria gap treino/inferência.
    Chame ANTES de mover o modelo para o device e de criar o optimizer.
    """
    from torchvision.ops.misc import FrozenBatchNorm2d

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            frozen.weight.data.copy_(child.weight.data)
            frozen.bias.data.copy_(child.bias.data)
            frozen.running_mean.data.copy_(child.running_mean.data)
            frozen.running_var.data.copy_(child.running_var.data)
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)
    return module
