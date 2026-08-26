"""Encoder de texto congelado, compartilhado pelos 3 mecanismos de fusão.

Escolha: sentence-transformers, não CLIP. As instruções do LIBERO são frases
templated simples ("pick up the X and place it in the basket") -- não
precisamos da capacidade (nem do peso) de um encoder multimodal como o CLIP,
e não usaríamos a torre de imagem dele mesmo (o backbone visual já é nosso).
all-MiniLM-L6-v2 é pequeno (~22M params, 384-dim), roda rápido até em CPU.
"""

import functools

import torch

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEXT_EMBED_DIM = 384  # dimensão nativa do all-MiniLM-L6-v2


@functools.lru_cache(maxsize=4)
def _load_frozen_sentence_transformer(model_name: str):
    """Cache em nível de módulo: se mais de um LanguageFusion pedir o mesmo
    model_name (ex: comparando os 3 mecanismos no mesmo processo), carrega
    o modelo da HuggingFace só uma vez."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class TextEmbeddingCache:
    """Codifica texto com um sentence-transformer congelado, cacheando por
    string exata -- as instruções do LIBERO são um vocabulário pequeno e
    fixo (~10-40 strings por fase), não faz sentido recomputar a cada batch.

    Fica FORA da árvore de nn.Module de propósito: é congelado e
    determinístico, não deveria inflar checkpoints com uma cópia do encoder
    de texto a cada salvamento -- só a projeção treinável (dentro de cada
    mecanismo de fusão, que USA esta classe por composição) precisa ser
    salva no state_dict.

    Lazy-load: o modelo só baixa/carrega no primeiro `encode()` de verdade,
    não na construção -- evita custo se o objeto for criado mas não usado.
    """

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL):
        self.model_name = model_name
        self._st_model = None
        self._cache: dict[str, torch.Tensor] = {}

    def _ensure_loaded(self):
        if self._st_model is None:
            self._st_model = _load_frozen_sentence_transformer(self.model_name)

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """(lista de B strings) -> (B, embed_dim), sem gradiente (congelado)."""
        self._ensure_loaded()
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            embs = self._st_model.encode(uncached, convert_to_tensor=True)
            for t, e in zip(uncached, embs):
                self._cache[t] = e.detach().cpu()  # cache em CPU; move pro device no uso
        return torch.stack([self._cache[t] for t in texts]).to(device)
