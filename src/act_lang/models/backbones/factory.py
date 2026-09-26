"""Factory: constrói o backbone visual certo a partir de uma string do config,
e resolve o par de text encoder correspondente (mesmo checkpoint original
pras duas modalidades, quando o backbone não é o ResNet18 original).

Mesmo padrão já usado por fusion/factory.py (build_fusion) -- trocar de
backbone é UMA linha no config (`backbone_type`), sem tocar em código.
"""

from ..fusion.text_encoder import DEFAULT_TEXT_MODEL, TextEmbeddingCache
from ..fusion.text_encoder_clip import CLIPTextEmbeddingCache
from ..fusion.text_encoder_siglip import SiglipTextEmbeddingCache
from .clip_vit import DEFAULT_CLIP_MODEL, CLIPVisionBackbone
from .resnet import VisionBackbone
from .siglip_vit import DEFAULT_SIGLIP_MODEL, SiglipVisionBackbone

BACKBONE_REGISTRY = {
    "resnet18": {
        "vision_cls": VisionBackbone,
        "vision_kwargs": {},
        "text_cls": TextEmbeddingCache,
        "text_model_name": DEFAULT_TEXT_MODEL,
    },
    "clip_vitb32": {
        "vision_cls": CLIPVisionBackbone,
        "vision_kwargs": {"model_name": DEFAULT_CLIP_MODEL},
        "text_cls": CLIPTextEmbeddingCache,
        "text_model_name": DEFAULT_CLIP_MODEL,
    },
    "siglip2_base": {
        "vision_cls": SiglipVisionBackbone,
        "vision_kwargs": {"model_name": DEFAULT_SIGLIP_MODEL},
        "text_cls": SiglipTextEmbeddingCache,
        "text_model_name": DEFAULT_SIGLIP_MODEL,
    },
}


def build_vision_backbone(backbone_type: str, d_model: int, pretrained: bool = True):
    """backbone_type: "resnet18" (default) | "clip_vitb32" | "siglip2_base"."""
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"backbone_type={backbone_type!r} desconhecido. "
            f"Opções: {sorted(BACKBONE_REGISTRY)}."
        )
    entry = BACKBONE_REGISTRY[backbone_type]
    return entry["vision_cls"](d_model=d_model, pretrained=pretrained, **entry["vision_kwargs"])


def resolve_text_encoder_spec(backbone_type: str):
    """(classe do text encoder pareado, model_name) -- pra build_fusion()
    injetar o text encoder certo e descobrir o embed_dim certo (lido de
    model.config.hidden_size na hora do load, nunca hardcoded no config)."""
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"backbone_type={backbone_type!r} desconhecido. "
            f"Opções: {sorted(BACKBONE_REGISTRY)}."
        )
    entry = BACKBONE_REGISTRY[backbone_type]
    return entry["text_cls"], entry["text_model_name"]
