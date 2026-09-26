"""Encoder de texto congelado: torre de texto do SigLIP 2
(google/siglip2-base-patch16-224), pareada com SiglipVisionBackbone
(models/backbones/siglip_vit.py) -- mesmo checkpoint original pras duas
modalidades.

Implementa a MESMA interface pública de TextEmbeddingCache (text_encoder.py):
`encode(texts, device) -> (B, embed_dim)` e `encode_tokens(texts, device) ->
(tokens, key_padding_mask)`.

Diferença de implementação que exige atenção (confirmada rodando ao vivo
contra o checkpoint real, não assumida): o tokenizer do SigLIP **NÃO
devolve `attention_mask`** -- ao contrário do CLIP (ver text_encoder_clip.py),
o protocolo do SigLIP sempre preenche até um comprimento FIXO (`max_length`,
default 64 abaixo -- instruções do LIBERO são bem mais curtas que isso) com
o token de padding (`tokenizer.pad_token_id`, confirmado = 0 pro checkpoint
usado), sem produzir máscara nenhuma -- o modelo foi pré-treinado assim,
sem mascarar atenção sobre o padding. Pra alimentar CrossAttentionFusion
(que precisa saber quais posições são padding de verdade), construímos a
`key_padding_mask` NÓS MESMOS comparando `input_ids == pad_token_id`. Isso
é uma decisão de engenharia deste projeto, não algo herdado do SigLIP.

`embed_dim` lido de `model.config.hidden_size` no load, nunca hardcoded.
"""

import functools

import torch

DEFAULT_SIGLIP_TEXT_MODEL = "google/siglip2-base-patch16-224"
SIGLIP_MAX_TEXT_LEN = 64  # convenção do próprio SigLIP; instruções do LIBERO são bem mais curtas


@functools.lru_cache(maxsize=4)
def _load_frozen_siglip_text(model_name: str):
    from transformers import AutoTokenizer, SiglipTextModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SiglipTextModel.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return tokenizer, model


class SiglipTextEmbeddingCache:
    """Mesmo contrato de TextEmbeddingCache (text_encoder.py), mas com a
    torre de texto do SigLIP -- ver docstring do módulo."""

    def __init__(self, model_name: str = DEFAULT_SIGLIP_TEXT_MODEL):
        from transformers import SiglipTextConfig

        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        # Lido AGORA (config é um download pequeno, não os pesos completos) --
        # ver comentário equivalente em text_encoder_clip.py.
        self.embed_dim: int = SiglipTextConfig.from_pretrained(model_name).hidden_size
        self._cache: dict[str, torch.Tensor] = {}
        self._token_cache: dict[str, torch.Tensor] = {}  # sequência JÁ recortada (sem padding)

    def _ensure_loaded(self):
        if self._model is None:
            self._tokenizer, self._model = _load_frozen_siglip_text(self.model_name)

    def _tokenize(self, texts: list[str]):
        return self._tokenizer(
            texts, padding="max_length", max_length=SIGLIP_MAX_TEXT_LEN,
            truncation=True, return_tensors="pt",
        )

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """(lista de B strings) -> (B, embed_dim), sem gradiente."""
        self._ensure_loaded()
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            inputs = self._tokenize(uncached)
            pooled = self._model(input_ids=inputs["input_ids"]).pooler_output  # (n_uncached, embed_dim)
            for t, e in zip(uncached, pooled):
                self._cache[t] = e.detach().cpu()
        return torch.stack([self._cache[t] for t in texts]).to(device)

    @torch.no_grad()
    def encode_tokens(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(lista de B strings) -> (tokens (B,L_max,embed_dim), key_padding_mask (B,L_max) bool True=padding).

        L_max aqui é sempre SIGLIP_MAX_TEXT_LEN (o SigLIP não trunca por
        frase -- todas as sequências já saem do tokenizer com esse
        comprimento fixo), diferente do MiniLM/CLIP, onde L_max é o maior
        comprimento REAL do batch.
        """
        self._ensure_loaded()
        uncached = [t for t in texts if t not in self._token_cache]
        if uncached:
            inputs = self._tokenize(uncached)
            hidden = self._model(input_ids=inputs["input_ids"]).last_hidden_state  # (n_uncached, L, embed_dim)
            pad_mask = inputs["input_ids"] == self._tokenizer.pad_token_id  # True = padding (construída à mão)
            for i, t in enumerate(uncached):
                length = int((~pad_mask[i]).sum().item())  # recorta até o último token não-padding
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
