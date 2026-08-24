"""Decoder estilo DETR — reinjeção posicional em CADA camada.

Diferença em relação ao nn.TransformerDecoder padrão (usado quando
decoder_style="torch" em ACT): aqui `query_pos` e `memory_pos` são somados a
q e k em toda camada (nunca a v), e `tgt` começa em zero — `action_queries`
passa a ser *apenas* posição, não conteúdo+posição fundidos.

Ablação isolada: liga com `ACT(..., decoder_style="detr")`. O baseline
("torch") não é afetado por este arquivo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _with_pos(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


class DETRStyleDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # self-attention entre as queries: pos em q/k, v é o conteúdo cru
        q = k = _with_pos(tgt, query_pos)
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(q, k, value=tgt)[0]))

        # cross-attention pro memory: query_pos no q, memory_pos no k, v cru
        q = _with_pos(tgt, query_pos)
        k = _with_pos(memory, memory_pos)
        tgt = self.norm2(tgt + self.dropout2(self.cross_attn(q, k, value=memory)[0]))

        tgt2 = self.linear2(self.dropout_ffn(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class DETRStyleDecoder(nn.Module):
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
                DETRStyleDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, query_pos, memory_pos)
        return tgt
