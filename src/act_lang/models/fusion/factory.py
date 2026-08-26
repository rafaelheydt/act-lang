"""Factory: constrói o LanguageFusion certo a partir de uma string do config.

Permite trocar de mecanismo com UMA linha no config (`fusion_type`), sem
tocar no notebook nem no código do modelo -- o mesmo padrão já usado por
`decoder_style` e `val_strategy`.
"""

from .base import LanguageFusion
from .cross_attn import CrossAttentionFusion
from .film import FiLMFusion
from .token import TokenFusion

FUSION_REGISTRY = {
    "token": TokenFusion,
    "film": FiLMFusion,
    "cross_attn": CrossAttentionFusion,
}


def build_fusion(fusion_type: str | None, d_model: int, **kwargs) -> LanguageFusion | None:
    """fusion_type: None (baseline, sem linguagem) | "token" | "film" | "cross_attn".

    kwargs extras (ex: n_heads=8 para "cross_attn") são repassados direto
    pro construtor do mecanismo escolhido -- só passe kwargs que esse
    mecanismo específico aceita (senão dá TypeError, o que é intencional:
    melhor falhar alto do que silenciosamente ignorar um parâmetro).
    """
    if fusion_type is None:
        return None
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(
            f"fusion_type={fusion_type!r} desconhecido. "
            f"Opções: {sorted(FUSION_REGISTRY)} ou None."
        )
    return FUSION_REGISTRY[fusion_type](d_model=d_model, **kwargs)
