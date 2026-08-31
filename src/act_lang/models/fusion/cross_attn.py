"""Fusão por CROSS-ATTENTION: cada token de observação "pergunta" à
instrução o que é relevante, via um bloco de atenção cruzada residual,
antes do transformer encoder principal processar tudo junto.

O mais expressivo dos três: diferente do FiLM (mesma modulação gamma/beta
pra todos os tokens) e do TokenFusion (encoder aprende a usar 1 token
extra por conta própria), aqui cada token de observação pode atender à
linguagem de forma DIFERENTE dos demais -- o token da imagem perto da
cesta pode "prestar mais atenção" em "basket" enquanto o token sobre o
leite atende a "milk". `fuse()` preserva o número de tokens (N -> N),
como o FiLM.

CORREÇÃO (set/2026) -- degenerescência com key única: a versão anterior
usava o embedding POOLED da frase como key/value único. Atenção sobre 1
key é softmax de 1 logito = 1.0 sempre, independente da query -> o
"attended" saía IDÊNTICO para todos os tokens de observação, colapsando o
mecanismo num deslocamento aditivo uniforme (um FiLM só-de-beta, mais
caro). Agora a instrução entra como sequência de embeddings POR TOKEN
(via TextEmbeddingCache.encode_tokens), com key_padding_mask para os
comprimentos variados do batch -- a atenção volta a ter sobre o que
distribuir peso, que é a hipótese que este mecanismo existe pra testar.
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

    def encode_text(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (lang_tokens, key_padding_mask).

        lang_tokens: (B, L, d_model) -- um token de linguagem por token da
        frase, projetado pra d_model. key_padding_mask: (B, L) bool, True
        onde é padding (convenção do nn.MultiheadAttention). O par viaja
        junto até o fuse() -- o ACT trata o retorno como objeto opaco.
        """
        raw, pad_mask = self._text_encoder.encode_tokens(texts, device)  # sem gradiente
        return self.proj(raw), pad_mask  # projeção treinável, (B, L, d_model)

    def attend(
        self, obs_tokens: torch.Tensor, lang: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Só a leitura da linguagem (sem residual/norm) -- isolado para
        diagnóstico e teste: com L>1 keys, a saída DEVE variar entre tokens
        de observação com conteúdos distintos (não-degenerescência)."""
        lang_tokens, pad_mask = lang
        attended, _ = self.cross_attn(
            query=obs_tokens, key=lang_tokens, value=lang_tokens,
            key_padding_mask=pad_mask,
        )
        return attended

    def fuse(
        self, obs_tokens: torch.Tensor, lang: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return self.norm(obs_tokens + self.attend(obs_tokens, lang))  # residual + norm
