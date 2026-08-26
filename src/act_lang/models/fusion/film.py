"""Fusão FiLM (Perez et al., 2018): linguagem modula os tokens de observação
via escala + deslocamento (gamma, beta) -- multiplicativo/aditivo, não
concatenação nem atenção.

Diferença estrutural dos outros dois mecanismos: `fuse()` preserva o número
de tokens (N -> N, ao contrário do TokenFusion que adiciona 1). A
informação de linguagem entra RE-ESCALANDO e DESLOCANDO o conteúdo já
existente, de forma idêntica pra todos os tokens do batch item (gamma/beta
são vetores (d_model,), broadcast sobre a dimensão de tokens).

Init em zero: `to_gamma_beta` começa com pesos e bias zerados, então no
início do treino gamma=0 e beta=0 -> fuse(x) = x*(1+0)+0 = x (identidade
exata). O modelo aprende a se afastar disso gradualmente, em vez de começar
com uma transformação aleatória destrutiva.
"""

import torch
import torch.nn as nn

from .base import LanguageFusion
from .text_encoder import DEFAULT_TEXT_EMBED_DIM, DEFAULT_TEXT_MODEL, TextEmbeddingCache


class FiLMFusion(LanguageFusion):
    def __init__(
        self,
        d_model: int,
        text_model_name: str = DEFAULT_TEXT_MODEL,
        text_embed_dim: int = DEFAULT_TEXT_EMBED_DIM,
    ):
        super().__init__()
        self._text_encoder = TextEmbeddingCache(text_model_name)
        self.to_gamma_beta = nn.Linear(text_embed_dim, 2 * d_model)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.d_model = d_model

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        raw = self._text_encoder.encode(texts, device)  # (B, text_embed_dim)
        return self.to_gamma_beta(raw)  # (B, 2*d_model)

    def fuse(self, obs_tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        gamma, beta = lang.split(self.d_model, dim=-1)  # (B, d_model) cada
        gamma = gamma.unsqueeze(1)  # (B, 1, d_model) -- broadcast pra todos os tokens
        beta = beta.unsqueeze(1)
        return obs_tokens * (1 + gamma) + beta
