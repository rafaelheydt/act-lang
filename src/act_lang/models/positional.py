"""Codificações posicionais usadas pelo ACT (1D para sequências, 2D para feature maps)."""

import math

import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    """Sinusoidal clássica (Vaswani et al.) para sequências de tokens."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def build_2d_sincos_position_embedding(
    h: int, w: int, d_model: int, device: torch.device
) -> torch.Tensor:
    """Embedding 2D sincos para os tokens espaciais do backbone visual.

    Retorna (1, h*w, d_model), pronto para somar aos tokens achatados.
    """
    assert d_model % 4 == 0, "d_model precisa ser múltiplo de 4 para o sincos 2D"
    eps = 1e-6
    scale = 2 * math.pi
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device) + 1,   # começa em 1, não em 0
        torch.arange(w, dtype=torch.float32, device=device) + 1,
        indexing="ij",
    )
    grid_y = grid_y / (h + eps) * scale   # normaliza pra faixa (0, 2π]
    grid_x = grid_x / (w + eps) * scale
    dim_quarter = d_model // 4
    omega = torch.exp(torch.arange(dim_quarter, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim_quarter))
    out_x = grid_x.flatten()[:, None] * omega[None, :]
    out_y = grid_y.flatten()[:, None] * omega[None, :]
    pe = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1
    )
    return pe.unsqueeze(0)
