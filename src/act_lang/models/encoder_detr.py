"""Encoder estilo DETR — reinjeção posicional em CADA camada.

Diferença em relação ao nn.TransformerEncoder padrão (usado tanto no encoder
principal do ACT quanto no CVAEEncoder, quando decoder_style="torch"): aqui
`pos` é somado a q e k em toda camada de self-attention (nunca a v) — mesmo
padrão do decoder_detr.py, só que sem cross-attention (o encoder não tem
memory externa, só os próprios tokens de entrada).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _with_pos(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


class DETRStyleEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # self-attention entre os tokens: pos em q/k, v é o conteúdo cru
        q = k = _with_pos(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src + self.dropout1(src2))

        src2 = self.linear2(self.dropout_ffn(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class DETRStyleEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DETRStyleEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, pos, src_key_padding_mask)
        return src