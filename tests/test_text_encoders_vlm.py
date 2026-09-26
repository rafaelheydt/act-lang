"""Testes dos text encoders pareados CLIPTextEmbeddingCache/SiglipTextEmbeddingCache
(ablação "backbones pareados", ver docs/hdf5_migration.md).

Mesma filosofia de tests/test_fusion.py: monkeypatch no ponto de carregamento
(aqui, `_load_frozen_clip_text`/`_load_frozen_siglip_text`, e o
`.from_pretrained` de config usado pra descobrir `embed_dim`) -- sem isso,
baixaria tokenizer/modelo de verdade (rede). Tokenizers reais não dão pra
sintetizar localmente do jeito que um Config pequeno dá (vocabulário é um
artefato aprendido) -- por isso o mock aqui é de um tokenizer/modelo FAKE
inteiro, não de uma config sintética real como em test_vision_backbones.py.

O que se testa: a MECÂNICA de cada classe (embed_dim correto, cache por
string, e principalmente a inversão de máscara de padding -- convenção do
HF, 1=válido -- para a convenção do projeto, True=padding -- que é o ponto
de maior risco de bug silencioso apontado nos dois arquivos).
"""

import pytest
import torch

pytest.importorskip("transformers")

CLIP_EMBED_DIM = 8
SIGLIP_EMBED_DIM = 8
SIGLIP_MAX_LEN = 64
PAD_ID = 0


def _lengths_for(texts: list[str]) -> list[int]:
    """Comprimento determinístico por string (3..5 tokens), só pra variar
    entre frases e exercitar padding -- mesmo espírito do hash-based length
    já usado no fixture de test_fusion.py."""
    return [3 + (hash(t) % 3) for t in texts]


class _FakeBatchEncoding(dict):
    """input_ids/attention_mask acessíveis por chave OU por atributo
    (** e [] são o suficiente pro código de produção; BatchEncoding real
    da HF também suporta os dois, não precisa herdar dela de verdade)."""


class _FakeCLIPTokenizer:
    def __call__(self, texts, padding=True, return_tensors="pt"):
        lengths = _lengths_for(texts)
        l_max = max(lengths)
        input_ids = torch.zeros(len(texts), l_max, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), l_max, dtype=torch.long)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = 1
            attention_mask[i, :length] = 1
        return _FakeBatchEncoding(input_ids=input_ids, attention_mask=attention_mask)


class _FakeModelOutput:
    def __init__(self, pooler_output, last_hidden_state):
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state


class _FakeCLIPTextModel:
    def __call__(self, input_ids, attention_mask=None):
        b, l = input_ids.shape
        g = torch.Generator().manual_seed(int(input_ids.sum().item()) % (2**31) + 1)
        hidden = torch.rand(b, l, CLIP_EMBED_DIM, generator=g)
        pooled = hidden[:, 0, :]
        return _FakeModelOutput(pooled, hidden)


class _FakeSiglipTokenizer:
    pad_token_id = PAD_ID

    def __call__(self, texts, padding="max_length", max_length=SIGLIP_MAX_LEN, truncation=True, return_tensors="pt"):
        lengths = _lengths_for(texts)
        input_ids = torch.full((len(texts), max_length), PAD_ID, dtype=torch.long)
        for i, length in enumerate(lengths):
            input_ids[i, :length] = 1  # token "real" -- qualquer id != PAD_ID
        return _FakeBatchEncoding(input_ids=input_ids)


class _FakeSiglipTextModel:
    def __call__(self, input_ids):
        b, l = input_ids.shape
        g = torch.Generator().manual_seed(int(input_ids.sum().item()) % (2**31) + 1)
        hidden = torch.rand(b, l, SIGLIP_EMBED_DIM, generator=g)
        pooled = hidden[:, 0, :]
        return _FakeModelOutput(pooled, hidden)


TASK_TEXTS = [
    "pick up the milk and place it in the basket",
    "pick up the cheese and place it in the basket",
]


@pytest.fixture(autouse=True)
def sem_download_de_verdade(monkeypatch):
    from transformers import CLIPTextConfig, SiglipTextConfig

    clip_config = CLIPTextConfig(hidden_size=CLIP_EMBED_DIM, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2)
    siglip_config = SiglipTextConfig(hidden_size=SIGLIP_EMBED_DIM, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2)
    monkeypatch.setattr(CLIPTextConfig, "from_pretrained", classmethod(lambda cls, name: clip_config))
    monkeypatch.setattr(SiglipTextConfig, "from_pretrained", classmethod(lambda cls, name: siglip_config))

    import act_lang.models.fusion.text_encoder_clip as clip_mod
    import act_lang.models.fusion.text_encoder_siglip as siglip_mod

    monkeypatch.setattr(clip_mod, "_load_frozen_clip_text", lambda name: (_FakeCLIPTokenizer(), _FakeCLIPTextModel()))
    monkeypatch.setattr(siglip_mod, "_load_frozen_siglip_text", lambda name: (_FakeSiglipTokenizer(), _FakeSiglipTextModel()))


class TestCLIPTextEmbeddingCache:
    def test_embed_dim_correto_lido_da_config(self):
        from act_lang.models.fusion.text_encoder_clip import CLIPTextEmbeddingCache

        cache = CLIPTextEmbeddingCache()
        assert cache.embed_dim == CLIP_EMBED_DIM  # não é o default do MiniLM (384), nem 768 (torre de imagem)

    def test_encode_shape(self):
        from act_lang.models.fusion.text_encoder_clip import CLIPTextEmbeddingCache

        cache = CLIPTextEmbeddingCache()
        out = cache.encode(TASK_TEXTS, torch.device("cpu"))
        assert out.shape == (2, CLIP_EMBED_DIM)

    def test_encode_tokens_mascara_invertida_corretamente(self):
        """HF: attention_mask 1=válido. Projeto: key_padding_mask True=padding.
        Ponto de maior risco de bug silencioso (uma inversão trocada não
        quebra shape nenhum)."""
        from act_lang.models.fusion.text_encoder_clip import CLIPTextEmbeddingCache

        cache = CLIPTextEmbeddingCache()
        tokens, mask = cache.encode_tokens(TASK_TEXTS, torch.device("cpu"))

        lengths = _lengths_for(TASK_TEXTS)
        l_max = max(lengths)
        assert tokens.shape == (2, l_max, CLIP_EMBED_DIM)
        assert mask.shape == (2, l_max)
        assert mask.dtype == torch.bool
        for i, length in enumerate(lengths):
            assert not mask[i, :length].any(), "posições REAIS não deveriam estar marcadas como padding"
            assert mask[i, length:].all(), "posições de padding deveriam estar marcadas como True"

    def test_cache_por_string_evita_recomputar(self):
        from act_lang.models.fusion.text_encoder_clip import CLIPTextEmbeddingCache

        cache = CLIPTextEmbeddingCache()
        out1 = cache.encode(TASK_TEXTS[:1], torch.device("cpu"))
        out2 = cache.encode(TASK_TEXTS[:1], torch.device("cpu"))
        assert torch.equal(out1, out2)


class TestSiglipTextEmbeddingCache:
    def test_embed_dim_correto_lido_da_config(self):
        from act_lang.models.fusion.text_encoder_siglip import SiglipTextEmbeddingCache

        cache = SiglipTextEmbeddingCache()
        assert cache.embed_dim == SIGLIP_EMBED_DIM

    def test_encode_shape(self):
        from act_lang.models.fusion.text_encoder_siglip import SiglipTextEmbeddingCache

        cache = SiglipTextEmbeddingCache()
        out = cache.encode(TASK_TEXTS, torch.device("cpu"))
        assert out.shape == (2, SIGLIP_EMBED_DIM)

    def test_encode_tokens_mascara_construida_a_partir_do_pad_token_id(self):
        """SigLIP não devolve attention_mask (confirmado ao vivo contra o
        checkpoint real) -- a máscara é construída comparando input_ids
        contra pad_token_id. Este teste trava esse comportamento."""
        from act_lang.models.fusion.text_encoder_siglip import SiglipTextEmbeddingCache

        cache = SiglipTextEmbeddingCache()
        tokens, mask = cache.encode_tokens(TASK_TEXTS, torch.device("cpu"))

        lengths = _lengths_for(TASK_TEXTS)
        l_max = max(lengths)  # cache interno recorta até o último token real -- não fica em 64
        assert tokens.shape == (2, l_max, SIGLIP_EMBED_DIM)
        assert mask.shape == (2, l_max)
        for i, length in enumerate(lengths):
            assert not mask[i, :length].any()
            assert mask[i, length:].all()


class TestBuildFusionComTextEncoderInjetado:
    """Integração: build_fusion aceitando text_encoder/text_embed_dim
    injetados -- mesmo caminho que scripts/train.py::build_model_and_optimizer
    usa pra plugar CLIP/SigLIP nos 3 mecanismos de fusão SEM que eles saibam
    a diferença."""

    def test_film_com_clip_text_encoder(self):
        from act_lang.models.fusion import build_fusion
        from act_lang.models.fusion.text_encoder_clip import CLIPTextEmbeddingCache

        text_encoder = CLIPTextEmbeddingCache()
        fusion = build_fusion(
            "film", d_model=16, text_encoder=text_encoder, text_embed_dim=text_encoder.embed_dim,
        )
        assert fusion._text_encoder is text_encoder  # reusa a MESMA instância, não cria outra por dentro

        obs_tokens = torch.rand(2, 5, 16)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)
        assert fused.shape == obs_tokens.shape

    def test_cross_attn_com_siglip_text_encoder(self):
        from act_lang.models.fusion import build_fusion
        from act_lang.models.fusion.text_encoder_siglip import SiglipTextEmbeddingCache

        text_encoder = SiglipTextEmbeddingCache()
        fusion = build_fusion(
            "cross_attn", d_model=16, n_heads=4,
            text_encoder=text_encoder, text_embed_dim=text_encoder.embed_dim,
        )

        obs_tokens = torch.rand(2, 5, 16)
        lang = fusion.encode_text(TASK_TEXTS, torch.device("cpu"))
        fused = fusion.fuse(obs_tokens, lang)
        assert fused.shape == obs_tokens.shape

    def test_sem_text_encoder_injetado_continua_default_minilm(self):
        """Não-regressão: omitir text_encoder/text_embed_dim continua dando
        o comportamento original (TextEmbeddingCache/MiniLM), sem quebrar
        os 3 arquivos de mecanismo pros usos já existentes."""
        from act_lang.models.fusion import build_fusion
        from act_lang.models.fusion.text_encoder import TextEmbeddingCache

        fusion = build_fusion("film", d_model=16)
        assert isinstance(fusion._text_encoder, TextEmbeddingCache)
