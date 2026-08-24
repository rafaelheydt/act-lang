"""ACT (Action Chunking Transformer) com CVAE — Zhao et al., 2023.

CORREÇÕES aplicadas em relação ao notebook original (revisão de agosto/2026):
  1. CVAEEncoder recebe `is_pad` e mascara as ações de padding via
     `src_key_padding_mask` (fidelidade ao ACT original; sem isso o [CLS]
     resume ações repetidas artificialmente no fim dos episódios).
  2. Tokens de state e z ganham embeddings posicionais próprios
     (`extra_pos_embed`), equivalente ao `additional_pos_embed` do original.
  3. `forward(..., sample_posterior=False)` permite validação determinística
     com z = mu, separando ruído de amostragem do sinal de val_loss.
  4. Normalização ImageNet embutida no VisionBackbone (ver backbone.py).

Ponto de extensão (Fase 2): `fusion` recebe um LanguageFusion; com None o
modelo é o baseline sem linguagem, byte a byte.
"""

from typing import Optional

import torch
import torch.nn as nn

from .backbone import VisionBackbone
from .fusion.base import LanguageFusion
from .positional import PositionalEncoding1D, build_2d_sincos_position_embedding


class CVAEEncoder(nn.Module):
    """Posterior q(z | state, ações) — só existe no treino."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        d_model: int,
        latent_dim: int,
        chunk_size: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=chunk_size + 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=3200, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        state_tok = self.state_proj(state).unsqueeze(1)
        action_toks = self.action_proj(actions)
        tokens = torch.cat([cls, state_tok, action_toks], dim=1)
        tokens = self.pos_encoding(tokens)

        # CORREÇÃO 1: ações de padding não participam da atenção.
        key_padding_mask = None
        if is_pad is not None:
            prefix = torch.zeros(  # [CLS] e state nunca são padding
                batch_size, 2, dtype=torch.bool, device=actions.device
            )
            key_padding_mask = torch.cat([prefix, is_pad], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.to_mu(encoded[:, 0]), self.to_logvar(encoded[:, 0])


class ACT(nn.Module):
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        d_model: int = 512,
        latent_dim: int = 32,
        chunk_size: int = 50,
        n_cameras: int = 2,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        fusion: Optional[LanguageFusion] = None,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_cameras = n_cameras
        self.fusion = fusion  # None = baseline sem linguagem

        self.vision_backbone = VisionBackbone(d_model, pretrained=pretrained_backbone)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cvae_encoder = CVAEEncoder(
            action_dim, state_dim, d_model, latent_dim, chunk_size,
            n_heads=n_heads, dropout=dropout,
        )

        # CORREÇÃO 2: posições dedicadas para os tokens de state e z
        # (equivalente ao additional_pos_embed do ACT original).
        self.extra_pos_embed = nn.Parameter(torch.zeros(1, 2, d_model))
        nn.init.normal_(self.extra_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=3200, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.action_queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=3200, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoder_layers
        )
        self.action_head = nn.Linear(d_model, action_dim)

    def encode_observations(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        z: torch.Tensor,
        task_texts: Optional[list[str]] = None,
    ) -> torch.Tensor:
        camera_tokens = []
        for cam_idx in range(self.n_cameras):
            feat_map = self.vision_backbone(images[:, cam_idx])
            b, c, h, w = feat_map.shape
            pos_emb = build_2d_sincos_position_embedding(h, w, c, feat_map.device)
            tokens = feat_map.flatten(2).transpose(1, 2) + pos_emb
            camera_tokens.append(tokens)
        image_tokens = torch.cat(camera_tokens, dim=1)

        state_tok = self.state_proj(state).unsqueeze(1) + self.extra_pos_embed[:, 0:1]
        z_tok = self.latent_proj(z).unsqueeze(1) + self.extra_pos_embed[:, 1:2]
        tokens = torch.cat([image_tokens, state_tok, z_tok], dim=1)

        # Fase 2: injeção de linguagem antes do transformer encoder.
        if self.fusion is not None and task_texts is not None:
            lang = self.fusion.encode_text(task_texts, device=tokens.device)
            tokens = self.fusion.fuse(tokens, lang)

        return self.transformer_encoder(tokens)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        task_texts: Optional[list[str]] = None,
        sample_posterior: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Treino: passe `actions` (e `is_pad`). Inferência: `actions=None` -> z = 0.

        CORREÇÃO 3: com `sample_posterior=False`, usa z = mu (validação
        determinística, sem ruído de reparametrização no val_loss).
        """
        batch_size = images.size(0)
        if actions is not None:
            mu, logvar = self.cvae_encoder(state, actions, is_pad)
            if sample_posterior:
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(std) * std
            else:
                z = mu
        else:
            mu = logvar = None
            z = torch.zeros(batch_size, self.latent_dim, device=images.device)

        memory = self.encode_observations(images, state, z, task_texts)
        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        return self.action_head(decoded), mu, logvar
