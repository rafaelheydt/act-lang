"""Normalização min-max -> [-1, 1], com proteção contra dimensões constantes.

CORREÇÃO aplicada: `clamp_min(eps)` no range evita divisão por zero se alguma
dimensão de estado/ação for constante no dataset (NaN silencioso ao trocar de
suíte/stats).

Decisão de projeto registrada: usamos as stats GLOBAIS do dataset lerobot/libero
(40 tarefas), mesmo treinando em subconjuntos. Isso mantém a normalização
idêntica entre todas as fases da dissertação (1 tarefa -> 10 tarefas -> com
linguagem), permitindo comparação limpa e reuso de checkpoints.
"""

from dataclasses import dataclass

import torch


@dataclass
class MinMaxNormalizer:
    x_min: torch.Tensor
    x_range: torch.Tensor  # já com clamp_min aplicado

    @classmethod
    def from_lerobot_stats(cls, stats: dict, key: str, eps: float = 1e-8):
        x_min = torch.as_tensor(stats[key]["min"], dtype=torch.float32)
        x_max = torch.as_tensor(stats[key]["max"], dtype=torch.float32)
        return cls(x_min=x_min, x_range=(x_max - x_min).clamp_min(eps))

    def to(self, device: torch.device) -> "MinMaxNormalizer":
        return MinMaxNormalizer(self.x_min.to(device), self.x_range.to(device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_min) / self.x_range * 2.0 - 1.0

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        return (x_norm + 1.0) / 2.0 * self.x_range + self.x_min
