"""Testes dos mecanismos de fusão de linguagem (Fase 3).

Todos monkeypatcham TextEmbeddingCache.encode -- sem isso, os testes
baixariam o sentence-transformer de verdade (rede, lento, e indisponível
neste ambiente de CI/sandbox de qualquer forma). O que se testa aqui é a
MECÂNICA de cada fuse() (contagem de tokens, gradiente fluindo pelas partes
treináveis), não a qualidade do encoder de texto em si -- isso só se
verifica rodando de verdade, com rede, num experimento real.
"""

import pytest
import torch

from act_lang.models.fusion.text_encoder import DEFAULT_TEXT_EMBED_DIM, TextEmbeddingCache


@pytest.fixture(autouse=True)
def sem_download_de_verdade(monkeypatch):
    """Troca o encode() real (baixaria o modelo da HF) por um stub
    determinístico -- todo teste deste arquivo usa isso automaticamente."""

    def fake_encode(self, texts, device):
        vecs = []
        for t in texts:
            g = torch.Generator().manual_seed(hash(t) % (2**31))
            vecs.append(torch.rand(DEFAULT_TEXT_EMBED_DIM, generator=g))
        return torch.stack(vecs).to(device)

    monkeypatch.setattr(TextEmbeddingCache, "encode", fake_encode)


TASK_TEXTS = [
    "pick up the milk and place it in the basket",
    "pick up the cheese and place it in the basket",
]


class TestTokenFusion:
    def test_adiciona_exatamente_1_token(self):
        from act_lang.models.fusion.token import TokenFusion

        fusion = TokenFusion(d_model=32)
        obs_tokens = torch.rand(2, 10, 32)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)

        assert fused.shape == (2, 11, 32)
        assert torch.equal(fused[:, :10], obs_tokens)  # tokens originais intactos

    def test_gradiente_flui_pela_projecao(self):
        from act_lang.models.fusion.token import TokenFusion

        torch.manual_seed(0)
        fusion = TokenFusion(d_model=16)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        lang.sum().backward()

        assert fusion.proj.weight.grad is not None
        assert fusion.proj.weight.grad.abs().sum() > 0


class TestFiLMFusion:
    def test_preserva_contagem_de_tokens(self):
        from act_lang.models.fusion.film import FiLMFusion

        fusion = FiLMFusion(d_model=32)
        obs_tokens = torch.rand(2, 10, 32)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)

        assert fused.shape == obs_tokens.shape

    def test_identidade_exata_na_inicializacao(self):
        """to_gamma_beta começa zerado -> gamma=0, beta=0 -> fuse(x) == x
        (identidade exata, não aproximada) -- garante que o treino começa
        sem uma transformação aleatória destrutiva."""
        from act_lang.models.fusion.film import FiLMFusion

        fusion = FiLMFusion(d_model=32)
        obs_tokens = torch.rand(2, 10, 32)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)

        assert torch.equal(fused, obs_tokens)

    def test_gradiente_flui_pelo_gerador_gamma_beta(self):
        from act_lang.models.fusion.film import FiLMFusion

        torch.manual_seed(0)
        fusion = FiLMFusion(d_model=16)
        obs_tokens = torch.rand(2, 5, 16)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)
        fused.sum().backward()

        assert fusion.to_gamma_beta.weight.grad is not None
        assert fusion.to_gamma_beta.weight.grad.abs().sum() > 0


class TestCrossAttentionFusion:
    def test_preserva_contagem_de_tokens(self):
        from act_lang.models.fusion.cross_attn import CrossAttentionFusion

        fusion = CrossAttentionFusion(d_model=32, n_heads=4)
        obs_tokens = torch.rand(2, 10, 32)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)

        assert fused.shape == obs_tokens.shape

    def test_gradiente_flui_pelas_partes_treinaveis(self):
        from act_lang.models.fusion.cross_attn import CrossAttentionFusion

        torch.manual_seed(0)
        fusion = CrossAttentionFusion(d_model=16, n_heads=2)
        obs_tokens = torch.rand(2, 5, 16)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)
        fused.sum().backward()

        assert fusion.proj.weight.grad is not None
        assert fusion.proj.weight.grad.abs().sum() > 0
        assert fusion.cross_attn.out_proj.weight.grad is not None


class TestFusionIntegrationComACT:
    """Cada mecanismo plugado de verdade no ACT -- forward, backward, e a
    combinação mais arriscada: TokenFusion (muda contagem de tokens) com
    decoder_style='detr' (que depende de memory_pos alinhado com memory --
    ver o padding de memory_pos em ACT.encode_observations)."""

    def _make_act(self, fusion, decoder_style="torch"):
        from act_lang.models.act import ACT

        return ACT(
            action_dim=7, state_dim=8, d_model=32, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=False, fusion=fusion, decoder_style=decoder_style,
        )

    def _forward_e_backward(self, model):
        torch.manual_seed(0)
        images = torch.rand(2, 2, 3, 32, 32)
        state = torch.rand(2, 8)
        actions = torch.rand(2, 4, 7)
        is_pad = torch.zeros(2, 4, dtype=torch.bool)
        pred, mu, logvar = model(
            images, state, actions=actions, is_pad=is_pad, task_texts=TASK_TEXTS
        )
        assert pred.shape == (2, 4, 7)
        (pred.sum() + mu.sum() + logvar.sum()).backward()
        return model

    def test_token_fusion_no_act(self):
        from act_lang.models.fusion.token import TokenFusion

        model = self._make_act(TokenFusion(d_model=32))
        model = self._forward_e_backward(model)
        assert model.fusion.proj.weight.grad is not None

    def test_film_fusion_no_act(self):
        from act_lang.models.fusion.film import FiLMFusion

        model = self._make_act(FiLMFusion(d_model=32))
        model = self._forward_e_backward(model)
        assert model.fusion.to_gamma_beta.weight.grad is not None

    def test_cross_attention_fusion_no_act(self):
        from act_lang.models.fusion.cross_attn import CrossAttentionFusion

        model = self._make_act(CrossAttentionFusion(d_model=32, n_heads=4))
        model = self._forward_e_backward(model)
        assert model.fusion.proj.weight.grad is not None

    def test_token_fusion_com_decoder_detr(self):
        """Caso mais arriscado: TokenFusion muda N -> N+1 tokens, e o
        decoder_style='detr' depende de memory_pos ter o MESMO tamanho de
        memory (pra reinjeção posicional no cross-attention) -- é o padding
        de memory_pos com zeros, em ACT.encode_observations, que evita
        isso quebrar."""
        from act_lang.models.fusion.token import TokenFusion

        model = self._make_act(TokenFusion(d_model=32), decoder_style="detr")
        self._forward_e_backward(model)  # não deveria levantar shape mismatch

    def test_fusion_none_continua_sendo_o_baseline(self):
        """Garantia de não-regressão: ACT(fusion=None) -- o caminho que
        todas as Fases 1 e 2 usam -- continua funcionando exatamente igual,
        sem nenhuma das classes deste arquivo interferindo."""
        model = self._make_act(fusion=None)
        self._forward_e_backward(model)


class TestBuildFusion:
    def test_none_devolve_none(self):
        from act_lang.models.fusion import build_fusion

        assert build_fusion(None, d_model=32) is None

    def test_constroi_cada_mecanismo_pelo_nome(self):
        from act_lang.models.fusion import build_fusion
        from act_lang.models.fusion.cross_attn import CrossAttentionFusion
        from act_lang.models.fusion.film import FiLMFusion
        from act_lang.models.fusion.token import TokenFusion

        assert isinstance(build_fusion("token", d_model=32), TokenFusion)
        assert isinstance(build_fusion("film", d_model=32), FiLMFusion)
        assert isinstance(build_fusion("cross_attn", d_model=32, n_heads=4), CrossAttentionFusion)

    def test_nome_invalido_da_erro_claro(self):
        from act_lang.models.fusion import build_fusion

        with pytest.raises(ValueError, match="fusion_type"):
            build_fusion("mecanismo_que_nao_existe", d_model=32)


class TestFluxoCompletoConfigAteForward:
    """Simula o que o notebook faz de verdade: cfg.get('fusion_type') ->
    build_fusion -> ACT(fusion=...) -> forward -- pros 3 mecanismos e pro
    caso baseline (fusion_type ausente do config, como nas Fases 1/2)."""

    @pytest.mark.parametrize("fusion_type", ["token", "film", "cross_attn", None])
    def test_config_ate_forward(self, fusion_type):
        from act_lang.models.act import ACT
        from act_lang.models.fusion import build_fusion

        cfg = {"d_model": 32}  # como cfg.get("fusion_type") quando ausente
        if fusion_type is not None:
            cfg["fusion_type"] = fusion_type

        fusion = build_fusion(cfg.get("fusion_type"), d_model=cfg["d_model"])
        model = ACT(
            action_dim=7, state_dim=8, d_model=32, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=False, fusion=fusion,
        )

        images = torch.rand(2, 2, 3, 32, 32)
        state = torch.rand(2, 8)
        actions = torch.rand(2, 4, 7)
        is_pad = torch.zeros(2, 4, dtype=torch.bool)
        pred, mu, logvar = model(
            images, state, actions=actions, is_pad=is_pad, task_texts=TASK_TEXTS
        )
        assert pred.shape == (2, 4, 7)
