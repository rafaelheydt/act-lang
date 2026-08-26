"""Fusão por TOKEN SIMPLES: linguagem vira +1 token, concatenado aos tokens
de observação antes do transformer encoder.

O mecanismo mais simples dos três -- não impõe nenhuma estrutura de ONDE ou
COMO a informação de linguagem deve influenciar a cena; o próprio
transformer encoder aprende isso via self-attention entre esse token e os
demais. `fuse()` muda o número de tokens (N -> N+1) -- é o único dos três
que faz isso (FiLM e cross-attention preservam N).
"""

import torch
import torch.nn as nn

from .base import LanguageFusion
from .text_encoder import DEFAULT_TEXT_EMBED_DIM, DEFAULT_TEXT_MODEL, TextEmbeddingCache


class TokenFusion(LanguageFusion):
    def __init__(
        self,
        d_model: int,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        text_embed_dim: int = DEFAULT_TEXT_EMBED_DIM,
    ):
        super().__init__()
        self._text_encoder = TextEmbeddingCache(text_model_name)
        self.proj = nn.Linear(text_embed_dim, d_model)  # única parte treinável

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        raw = self._text_encoder.encode(texts, device)  # (B, text_embed_dim), sem gradiente
        return self.proj(raw)  # (B, d_model)

    def fuse(self, obs_tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        lang_token = lang.unsqueeze(1)  # (B, 1, d_model)
        return torch.cat([obs_tokens, lang_token], dim=1)  # (B, N+1, d_model)
