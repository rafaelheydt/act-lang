"""Encoder de texto congelado: torre de texto do CLIP (openai/clip-vit-base-patch32),
pareada com CLIPVisionBackbone (models/backbones/clip_vit.py) -- mesmo
checkpoint original pras duas modalidades.

Implementa a MESMA interface pública de TextEmbeddingCache (text_encoder.py,
usada pelo MiniLM): `encode(texts, device) -> (B, embed_dim)` e
`encode_tokens(texts, device) -> (tokens, key_padding_mask)`. Os 3
mecanismos de fusão (film.py/token.py/cross_attn.py) não sabem nem precisam
saber qual classe concreta estão usando.

Diferença de implementação em relação ao MiniLM que exige atenção (não é
cópia mecânica): o `attention_mask` do tokenizer do `transformers` usa
convenção INVERTIDA (1=token válido) em relação ao `key_padding_mask` do
projeto (True=padding, convenção do nn.MultiheadAttention) -- invertida
explicitamente abaixo (`~attention_mask.bool()`). Uma inversão trocada não
quebra nenhum shape, só corrompe silenciosamente qual token cada mecanismo
de cross-attention atende -- coberto por teste dedicado.

`embed_dim` é lido de `model.config.hidden_size` no load, nunca hardcoded:
confirmado que a torre de TEXTO do CLIP ViT-B/32 tem hidden_size=512, não
768 (768 é a dimensão da torre de IMAGEM -- as duas torres do CLIP não
compartilham largura, só o espaço de projeção final, que este código não
usa: usamos os hidden states crus de cada torre, não os embeddings
projetados no espaço contrastivo compartilhado).
"""

import functools

import torch

from .text_encoder import DEFAULT_TEXT_EMBED_DIM  # noqa: F401 -- mantido por conveniência de import

DEFAULT_CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"


@functools.lru_cache(maxsize=4)
def _load_frozen_clip_text(model_name: str):
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return tokenizer, model


class CLIPTextEmbeddingCache:
    """Mesmo contrato de TextEmbeddingCache (text_encoder.py), mas com a
    torre de texto do CLIP -- ver docstring do módulo."""

    def __init__(self, model_name: str = DEFAULT_CLIP_TEXT_MODEL):
        from transformers import CLIPTextConfig

        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        # Lido AGORA (config é um download pequeno, não os pesos completos) --
        # quem constrói a fusão (build_fusion) precisa de embed_dim ANTES de
        # qualquer encode() de verdade acontecer, pra dimensionar a projeção
        # treinável (nn.Linear) do mecanismo. O modelo em si continua lazy
        # (só carrega no primeiro encode/encode_tokens de verdade).
        self.embed_dim: int = CLIPTextConfig.from_pretrained(model_name).hidden_size
        self._cache: dict[str, torch.Tensor] = {}
        self._token_cache: dict[str, torch.Tensor] = {}  # sequência JÁ recortada (sem padding), comprimento próprio por string

    def _ensure_loaded(self):
        if self._model is None:
            self._tokenizer, self._model = _load_frozen_clip_text(self.model_name)

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """(lista de B strings) -> (B, embed_dim), sem gradiente."""
        self._ensure_loaded()
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            inputs = self._tokenizer(uncached, padding=True, return_tensors="pt")
            pooled = self._model(**inputs).pooler_output  # (n_uncached, embed_dim)
            for t, e in zip(uncached, pooled):
                self._cache[t] = e.detach().cpu()
        return torch.stack([self._cache[t] for t in texts]).to(device)

    @torch.no_grad()
    def encode_tokens(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(lista de B strings) -> (tokens (B,L_max,embed_dim), key_padding_mask (B,L_max) bool True=padding)."""
        self._ensure_loaded()
        uncached = [t for t in texts if t not in self._token_cache]
        if uncached:
            inputs = self._tokenizer(uncached, padding=True, return_tensors="pt")
            hidden = self._model(**inputs).last_hidden_state  # (n_uncached, L, embed_dim)
            for i, t in enumerate(uncached):
                length = int(inputs["attention_mask"][i].sum().item())  # recorta o padding do próprio tokenizer
                self._token_cache[t] = hidden[i, :length].detach().cpu()

        seqs = [self._token_cache[t] for t in texts]
        lengths = [s.size(0) for s in seqs]
        l_max = max(lengths)
        tokens = torch.zeros(len(texts), l_max, self.embed_dim)
        mask = torch.ones(len(texts), l_max, dtype=torch.bool)  # True = padding
        for i, (s, length) in enumerate(zip(seqs, lengths)):
            tokens[i, :length] = s
            mask[i, :length] = False
        return tokens.to(device), mask.to(device)
