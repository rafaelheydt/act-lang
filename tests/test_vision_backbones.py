"""Testes dos backbones visuais pareados CLIP ViT-B/32 e SigLIP2-base
(ablação "backbones pareados", ver docs/hdf5_migration.md e
src/act_lang/models/backbones/).

Todos monkeypatcham as chamadas `.from_pretrained` da `transformers` --
sem isso, os testes baixariam checkpoints de verdade (rede, lento,
indisponível neste ambiente de CI/sandbox). Config sintética pequena
(hidden_size/image_size/patch_size minúsculos) construída localmente,
pesos aleatórios -- mesmo espírito de `pretrained_backbone=False` já usado
em TestFusionIntegrationComACT (test_fusion.py) pro ResNet18, aplicado ao ViT.
"""

import pytest
import torch

pytest.importorskip("transformers")


class _FakeImageProcessor:
    """Stub de AutoImageProcessor -- só os 2 atributos que os backbones leem."""

    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


IMAGE_SIZE = 32
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE  # 2 -> 4 patches
HIDDEN_SIZE = 8


@pytest.fixture(autouse=True)
def sem_download_de_verdade(monkeypatch):
    """Troca as chamadas .from_pretrained (baixariam checkpoints de verdade)
    por construções sintéticas locais -- pesos aleatórios, sem rede."""
    from transformers import (
        AutoImageProcessor, CLIPVisionConfig, CLIPVisionModel,
        SiglipVisionConfig, SiglipVisionModel,
    )

    clip_config = CLIPVisionConfig(
        hidden_size=HIDDEN_SIZE, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
    )
    siglip_config = SiglipVisionConfig(
        hidden_size=HIDDEN_SIZE, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
    )

    monkeypatch.setattr(CLIPVisionConfig, "from_pretrained", classmethod(lambda cls, name: clip_config))
    monkeypatch.setattr(CLIPVisionModel, "from_pretrained", classmethod(lambda cls, name: CLIPVisionModel(clip_config)))
    monkeypatch.setattr(SiglipVisionConfig, "from_pretrained", classmethod(lambda cls, name: siglip_config))
    monkeypatch.setattr(SiglipVisionModel, "from_pretrained", classmethod(lambda cls, name: SiglipVisionModel(siglip_config)))
    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", staticmethod(lambda name: _FakeImageProcessor()))


class TestCLIPVisionBackbone:
    def test_forward_shape_com_resize(self):
        """Imagem MENOR que image_size (64 no lugar de 32) -- exercita o
        F.interpolate dentro do forward (mesmo código que trata o pipeline
        HDF5 128x128 / JPEG 256x256 contra o tamanho nativo do checkpoint).
        Passar sem erro de shape já prova, de quebra, que o token CLS foi
        descartado antes do reshape (b, c, GRID, GRID) -- sem isso, o
        .reshape falharia (n_patches+1 não é quadrado perfeito)."""
        from act_lang.models.backbones.clip_vit import CLIPVisionBackbone

        backbone = CLIPVisionBackbone(d_model=12, pretrained=True)
        images = torch.rand(2, 3, 64, 64)
        out = backbone(images)

        assert out.shape == (2, 12, GRID, GRID)

    def test_pretrained_false_usa_config_sem_pesos_prontos(self):
        from act_lang.models.backbones.clip_vit import CLIPVisionBackbone

        backbone = CLIPVisionBackbone(d_model=12, pretrained=False)
        out = backbone(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert out.shape == (1, 12, GRID, GRID)

    def test_gradiente_flui_pela_projecao_quando_nao_congelado(self):
        from act_lang.models.backbones.clip_vit import CLIPVisionBackbone

        backbone = CLIPVisionBackbone(d_model=12, pretrained=True)
        out = backbone(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        out.sum().backward()

        assert backbone.proj.weight.grad is not None
        assert backbone.proj.weight.grad.abs().sum() > 0
        assert backbone.vision_model.embeddings.patch_embedding.weight.grad is not None
        # ^ CLIPVisionModel expõe embeddings/encoder direto (sem sub-atributo
        # .vision_model do CLIPModel completo) -- confirmado ao vivo com
        # config sintética antes de escrever este teste.

    def test_backbone_congelado_nao_acumula_gradiente(self):
        """Mesmo padrão que build_model_and_optimizer aplica quando
        cfg['freeze_vision_backbone'] é True (scripts/train.py). Com TODOS
        os parâmetros congelados, a saída não deveria sequer rastrear
        gradiente (nenhuma folha treinável no grafo) -- chamar .backward()
        nessa saída levantaria RuntimeError ("does not require grad"), o
        que já é a prova mais direta de que o congelamento funcionou."""
        from act_lang.models.backbones.clip_vit import CLIPVisionBackbone

        backbone = CLIPVisionBackbone(d_model=12, pretrained=True)
        for p in backbone.parameters():
            p.requires_grad = False

        out = backbone(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))

        assert out.requires_grad is False
        with pytest.raises(RuntimeError, match="does not require grad"):
            out.sum().backward()


class TestSiglipVisionBackbone:
    def test_forward_shape_sem_cls(self):
        from act_lang.models.backbones.siglip_vit import SiglipVisionBackbone

        backbone = SiglipVisionBackbone(d_model=12, pretrained=True)
        out = backbone(torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE))

        assert out.shape == (2, 12, GRID, GRID)

    def test_gradiente_flui_pela_projecao_quando_nao_congelado(self):
        from act_lang.models.backbones.siglip_vit import SiglipVisionBackbone

        backbone = SiglipVisionBackbone(d_model=12, pretrained=True)
        out = backbone(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        out.sum().backward()

        assert backbone.proj.weight.grad is not None
        assert backbone.proj.weight.grad.abs().sum() > 0


class TestFreezeBatchnormNoOp:
    def test_freeze_batchnorm_e_noop_seguro_em_backbones_vit(self):
        """freeze_batchnorm (models/backbones/resnet.py) é chamado
        incondicionalmente em scripts/train.py quando cfg['freeze_bn'] --
        precisa ser inofensivo em backbones sem nn.BatchNorm2d."""
        from act_lang.models.backbones.clip_vit import CLIPVisionBackbone
        from act_lang.models.backbones.resnet import freeze_batchnorm

        backbone = CLIPVisionBackbone(d_model=12, pretrained=True)
        state_before = {k: v.clone() for k, v in backbone.state_dict().items()}
        freeze_batchnorm(backbone)  # não deveria levantar nem mudar nada
        state_after = backbone.state_dict()

        assert set(state_before) == set(state_after)
        for k in state_before:
            assert torch.equal(state_before[k], state_after[k])


class TestBuildVisionBackboneFactory:
    def test_resolve_por_nome(self):
        from act_lang.models.backbones import build_vision_backbone
        from act_lang.models.backbones.clip_vit import CLIPVisionBackbone
        from act_lang.models.backbones.resnet import VisionBackbone
        from act_lang.models.backbones.siglip_vit import SiglipVisionBackbone

        assert isinstance(build_vision_backbone("resnet18", d_model=12, pretrained=False), VisionBackbone)
        assert isinstance(build_vision_backbone("clip_vitb32", d_model=12), CLIPVisionBackbone)
        assert isinstance(build_vision_backbone("siglip2_base", d_model=12), SiglipVisionBackbone)

    def test_nome_invalido_da_erro_claro(self):
        from act_lang.models.backbones import build_vision_backbone

        with pytest.raises(ValueError, match="backbone_type"):
            build_vision_backbone("backbone_que_nao_existe", d_model=12)

    def test_resolve_text_encoder_spec_pareado(self):
        from act_lang.models.backbones import resolve_text_encoder_spec
        from act_lang.models.fusion.text_encoder_clip import CLIPTextEmbeddingCache
        from act_lang.models.fusion.text_encoder_siglip import SiglipTextEmbeddingCache
        from act_lang.models.fusion.text_encoder import TextEmbeddingCache

        cls, name = resolve_text_encoder_spec("resnet18")
        assert cls is TextEmbeddingCache
        cls, name = resolve_text_encoder_spec("clip_vitb32")
        assert cls is CLIPTextEmbeddingCache
        cls, name = resolve_text_encoder_spec("siglip2_base")
        assert cls is SiglipTextEmbeddingCache


class TestACTComBackboneNovo:
    """Integração fim-a-fim: ACT(backbone_type=...) -- mesmo padrão de
    TestFusionIntegrationComACT (test_fusion.py), aplicado ao backbone
    novo em vez do mecanismo de fusão."""

    def _forward_e_backward(self, model):
        torch.manual_seed(0)
        images = torch.rand(2, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
        state = torch.rand(2, 8)
        actions = torch.rand(2, 4, 7)
        is_pad = torch.zeros(2, 4, dtype=torch.bool)
        pred, mu, logvar = model(images, state, actions=actions, is_pad=is_pad)
        assert pred.shape == (2, 4, 7)
        (pred.sum() + mu.sum() + logvar.sum()).backward()

    def test_clip_backbone_no_act(self):
        from act_lang.models.act import ACT

        model = ACT(
            action_dim=7, state_dim=8, d_model=12, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=True, backbone_type="clip_vitb32",
        )
        self._forward_e_backward(model)

    def test_siglip_backbone_no_act(self):
        from act_lang.models.act import ACT

        model = ACT(
            action_dim=7, state_dim=8, d_model=12, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=True, backbone_type="siglip2_base",
        )
        self._forward_e_backward(model)

    def test_freeze_vision_backbone_no_act_completo(self):
        """Cenário real de build_model_and_optimizer (scripts/train.py):
        só o backbone visual é congelado, o resto do modelo (aqui, sem
        fusão -- só pra isolar) continua treinável. Confirma que o
        gradiente flui pro resto do modelo mas NÃO pro backbone."""
        from act_lang.models.act import ACT

        model = ACT(
            action_dim=7, state_dim=8, d_model=12, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=True, backbone_type="clip_vitb32",
        )
        for p in model.vision_backbone.parameters():
            p.requires_grad = False

        self._forward_e_backward(model)

        assert model.vision_backbone.proj.weight.grad is None
        assert model.state_proj.weight.grad is not None  # resto do modelo continua treinável
        assert model.state_proj.weight.grad.abs().sum() > 0

    def test_backbone_type_default_continua_resnet18(self):
        """Não-regressão: omitir backbone_type continua dando o ResNet18
        original, byte a byte (mesmo caminho que todas as Fases anteriores
        usam)."""
        from act_lang.models.act import ACT
        from act_lang.models.backbones.resnet import VisionBackbone

        model = ACT(
            action_dim=7, state_dim=8, d_model=32, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=False,
        )
        assert isinstance(model.vision_backbone, VisionBackbone)
