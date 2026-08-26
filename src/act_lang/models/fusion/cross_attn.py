"""Fusão por CROSS-ATTENTION: cada token de observação "pergunta" à
instrução o que é relevante, via um bloco de atenção cruzada residual,
antes do transformer encoder principal processar tudo junto.

O mais expressivo dos três: diferente do FiLM (mesma modulação gamma/beta
pra todos os tokens) e do TokenFusion (encoder aprende a usar 1 token
extra por conta própria), aqui cada token pode atender à linguagem de
forma DIFERENTE dos demais -- o token da imagem perto da cesta pode
"prestar mais atenção" na instrução do que o token do canto vazio da mesa,
por exemplo. `fuse()` preserva o número de tokens (N -> N), como o FiLM.
"""

import torch
import torch.nn as nn

from .base import LanguageFusion
from .text_encoder import DEFAULT_TEXT_EMBED_DIM, DEFAULT_TEXT_MODEL, TextEmbeddingCache


class CrossAttentionFusion(LanguageFusion):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        text_embed_dim: int = DEFAULT_TEXT_EMBED_DIM,
    ):
        super().__init__()
        self._text_encoder = TextEmbeddingCache(text_model_name)
        self.proj = nn.Linear(text_embed_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        raw = self._text_encoder.encode(texts, device)  # (B, text_embed_dim)
        return self.proj(raw).unsqueeze(1)  # (B, 1, d_model) -- "memory" de 1 token

    def fuse(self, obs_tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attn(query=obs_tokens, key=lang, value=lang)
        return self.norm(obs_tokens + attended)  # residual + norm
