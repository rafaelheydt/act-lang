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

Ablação opcional (desligada por padrão): `decoder_style="detr"` troca o
nn.TransformerDecoder de prateleira (posição injetada 1x na entrada) por um
decoder que reinjeta query_pos/memory_pos em q,k de cada camada — fiel ao
DETR original. Default "torch" preserva o comportamento já validado; ver
decoder_detr.py.

Ponto de extensão (Fase 2): `fusion` recebe um LanguageFusion; com None o
modelo é o baseline sem linguagem, byte a byte.
"""

from typing import Optional

import torch
import torch.nn as nn

from .backbone import VisionBackbone
from .decoder_detr import DETRStyleDecoder
from .encoder_detr import DETRStyleEncoder
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
        attention_style: str = "torch",
    ):
        super().__init__()
        assert attention_style in ("torch", "detr"), attention_style
        self.attention_style = attention_style
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=chunk_size + 2)
        if attention_style == "torch":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=3200, dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:  # "detr": pos reinjetada em q/k a cada camada, nunca somada ao conteúdo
            self.encoder = DETRStyleEncoder(
                d_model, n_heads, n_layers, dim_feedforward=3200, dropout=dropout
            )
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

        # CORREÇÃO 1: ações de padding não participam da atenção.
        key_padding_mask = None
        if is_pad is not None:
            prefix = torch.zeros(  # [CLS] e state nunca são padding
                batch_size, 2, dtype=torch.bool, device=actions.device
            )
            key_padding_mask = torch.cat([prefix, is_pad], dim=1)

        if self.attention_style == "torch":
            tokens = self.pos_encoding(tokens)  # soma única, antes do encoder
            encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        else:  # "detr"
            pos = self.pos_encoding.pe[:, : tokens.size(1)]  # só os valores, sem somar
            encoded = self.encoder(tokens, pos=pos, src_key_padding_mask=key_padding_mask)

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
        decoder_style: str = "torch",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_cameras = n_cameras
        self.fusion = fusion  # None = baseline sem linguagem
        assert decoder_style in ("torch", "detr"), decoder_style
        self.decoder_style = decoder_style

        self.vision_backbone = VisionBackbone(d_model, pretrained=pretrained_backbone)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cvae_encoder = CVAEEncoder(
            action_dim, state_dim, d_model, latent_dim, chunk_size,
            n_heads=n_heads, dropout=dropout, attention_style=decoder_style,
        )

        # CORREÇÃO 2: posições dedicadas para os tokens de state e z
        # (equivalente ao additional_pos_embed do ACT original).
        self.extra_pos_embed = nn.Parameter(torch.zeros(1, 2, d_model))
        nn.init.normal_(self.extra_pos_embed, std=0.02)

        if decoder_style == "torch":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=3200, dropout=dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_encoder_layers
            )
        else:  # "detr": pos reinjetada em q/k a cada camada (ablação)
            self.transformer_encoder = DETRStyleEncoder(
                d_model, n_heads, n_encoder_layers, dim_feedforward=3200, dropout=dropout
            )

        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.action_queries, std=0.02)
        if decoder_style == "torch":
            decoder_layer = nn.TransformerDecoderLayer(
                d_model, n_heads, dim_feedforward=3200, dropout=dropout, batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=n_decoder_layers
            )
        else:  # "detr": reinjeção posicional em cada camada (ablação)
            self.transformer_decoder = DETRStyleDecoder(
                d_model, n_heads, n_decoder_layers, dim_feedforward=3200, dropout=dropout
            )
        self.action_head = nn.Linear(d_model, action_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier uniform no encoder/decoder principal, fiel ao
        Transformer._reset_parameters() oficial (DETR/ACT). NÃO toca no
        vision_backbone (pesos pré-treinados do ImageNet seriam destruídos)
        nem no cvae_encoder — o oficial também não reseta o encoder da CVAE,
        só o `Transformer` que faz o encode+decode principal.
        """
        for module in (self.transformer_encoder, self.transformer_decoder):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def encode_observations(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        z: torch.Tensor,
        task_texts: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (memory, memory_pos).

        Fusão de linguagem (Fase 2/3): aplicada SÓ nos tokens de imagem,
        antes de se juntarem a state/z — fiel à Figura 5 do MT-ACT
        (Bharadhwaj et al., 2023): a linguagem ajuda a imagem a focar no
        objeto/tarefa certa; state/z entram "limpos", sem essa modulação
        (eles não são ambíguos quanto à tarefa do jeito que a imagem é).

        Em decoder_style="torch": a posição é somada ao conteúdo (já com a
        fusão aplicada) uma única vez, porque o nn.TransformerEncoder de
        prateleira não aceita `pos` separado. `memory_pos` guarda uma CÓPIA
        dos mesmos embeddings já somados a `memory`, só para o
        decoder_style="detr" poder reinjetá-los no cross-attention do
        decoder sem duplicar a soma no conteúdo (v).

        Em decoder_style="detr": conteúdo e posição ficam separados até
        entrar no DETRStyleEncoder, que reinjeta `pos` em q/k a cada camada
        (nunca soma ao v) — fiel ao encoder principal do ACT/DETR oficial.
        """
        camera_tokens, camera_pos = [], []
        for cam_idx in range(self.n_cameras):
            feat_map = self.vision_backbone(images[:, cam_idx])
            b, c, h, w = feat_map.shape
            pos_emb = build_2d_sincos_position_embedding(h, w, c, feat_map.device)
            camera_tokens.append(feat_map.flatten(2).transpose(1, 2))  # conteúdo puro
            camera_pos.append(pos_emb.expand(b, -1, -1))
        image_content = torch.cat(camera_tokens, dim=1)
        image_pos = torch.cat(camera_pos, dim=1)
        batch_size = state.size(0)

        # Fusão de linguagem: só na imagem, antes de juntar com state/z.
        if self.fusion is not None and task_texts is not None:
            lang = self.fusion.encode_text(task_texts, device=image_content.device)
            image_content = self.fusion.fuse(image_content, lang)
            # alguns mecanismos (ex: TokenFusion) adicionam token(s) sem
            # posição definida -> pad com zeros pra image_pos continuar
            # alinhado token a token com image_content.
            if image_content.size(1) != image_pos.size(1):
                pad = torch.zeros(
                    batch_size, image_content.size(1) - image_pos.size(1), self.d_model,
                    device=image_content.device,
                )
                image_pos = torch.cat([image_pos, pad], dim=1)

        state_content = self.state_proj(state).unsqueeze(1)
        z_content = self.latent_proj(z).unsqueeze(1)
        content = torch.cat([image_content, state_content, z_content], dim=1)

        state_pos = self.extra_pos_embed[:, 0:1].expand(batch_size, -1, -1)
        z_pos = self.extra_pos_embed[:, 1:2].expand(batch_size, -1, -1)
        memory_pos = torch.cat([image_pos, state_pos, z_pos], dim=1)

        if self.decoder_style == "torch":
            memory = self.transformer_encoder(content + memory_pos)
        else:  # "detr"
            memory = self.transformer_encoder(content, pos=memory_pos)

        return memory, memory_pos

    def decode_with_z(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        z: torch.Tensor,
        task_texts: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Decodifica ações a partir de um `z` explícito, passando por cima
        do CVAEEncoder -- só para diagnóstico/inspeção (ex: testar
        sensibilidade do decoder a diferentes amostras de z ~ N(0,1), sem
        ficar restrito a z=mu ou z=0). Não é usado no treino nem na
        inferência normal -- `forward()` já cobre os dois casos reais e
        chama este método internamente.
        """
        batch_size = images.size(0)
        memory, memory_pos = self.encode_observations(images, state, z, task_texts)

        if self.decoder_style == "torch":
            # comportamento original: action_queries = conteúdo+posição
            # fundidos, injetados uma única vez como tgt.
            queries = self.action_queries.expand(batch_size, -1, -1)
            decoded = self.transformer_decoder(tgt=queries, memory=memory)
        else:  # "detr": tgt começa em zero; action_queries vira SÓ posição,
               # reinjetada (junto com memory_pos) em toda camada.
            query_pos = self.action_queries.expand(batch_size, -1, -1)
            tgt = torch.zeros_like(query_pos)
            decoded = self.transformer_decoder(
                tgt=tgt, memory=memory, query_pos=query_pos, memory_pos=memory_pos
            )

        return self.action_head(decoded)

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

        a_hat = self.decode_with_z(images, state, z, task_texts)
        return a_hat, mu, logvar