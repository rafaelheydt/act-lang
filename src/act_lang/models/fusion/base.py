"""Interface comum dos mecanismos de fusão de linguagem (Fase 2).

O ACT recebe um LanguageFusion opcional por injeção de dependência. Trocar o
mecanismo (token simples / FiLM / cross-attention) deve ser UMA linha no config
— nunca uma cópia do modelo. Cada mecanismo concreto vive em seu próprio
arquivo neste pacote e implementa esta interface.

Contrato:
  - `encode_text(texts)` transforma a lista de strings do batch em uma
    representação de linguagem (ex.: embeddings de um encoder congelado).
  - `fuse(obs_tokens, lang)` injeta a linguagem nos tokens de observação do
    transformer encoder e devolve os tokens (possivelmente com comprimento
    diferente, ex.: +1 token de linguagem no modo "token simples").

Enquanto a fusão for None no ACT (baseline sem linguagem), nada disso é usado.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LanguageFusion(nn.Module, ABC):
    @abstractmethod
    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """(lista de B strings) -> representação de linguagem, ex. (B, L, d) ou (B, d)."""

    @abstractmethod
    def fuse(self, obs_tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """(B, N, d_model), lang -> (B, N', d_model) com a linguagem incorporada."""
