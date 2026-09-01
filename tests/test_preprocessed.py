"""Testes do caminho de dados pré-processado e da gradient accumulation.

O que importa garantir aqui:
1. PreprocessedLiberoDataset honra o MESMO contrato de batch que o caminho
   lerobot (chaves, shapes, padding do chunk) -- é isso que deixa o
   LiberoActBridge e o resto do treino intactos.
2. accum_steps é matematicamente equivalente ao batch grande: N microbatches
   de tamanho B com accum N == 1 batch de tamanho N*B (mesmos pesos após o
   step), condição para o batch efetivo 32 ser comparável entre GPUs.
"""

import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from act_lang.data.libero import LiberoActBridge
from act_lang.data.normalize import MinMaxNormalizer
from act_lang.data.preprocessed import (
    ARRAYS_NAME, FRAMES_DIR, META_NAME, PREPROCESS_VERSION,
    PreprocessedLiberoDataset, build_normalizers_from_meta,
)

H, W = 16, 16
STATE_DIM, ACTION_DIM = 8, 7
TASK_A = "pick up the milk and place it in the basket"
TASK_B = "pick up the ketchup and place it in the basket"


@pytest.fixture()
def fake_root(tmp_path):
    """Dataset pré-processado sintético: 2 episódios (T=6 e T=4), 2 câmeras,
    PNG (sem perda -> os pixels lidos podem ser comparados com exatidão)."""
    rng = np.random.default_rng(0)
    episodes = {0: (6, TASK_A), 1: (4, TASK_B)}
    arrays, meta_eps = {}, {}
    for ep, (T, task) in episodes.items():
        ep_dir = tmp_path / FRAMES_DIR / f"ep{ep:05d}"
        ep_dir.mkdir(parents=True)
        for t in range(T):
            for cam in (0, 1):
                arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
                Image.fromarray(arr).save(ep_dir / f"c{cam}_{t:05d}.png")
        arrays[ep] = {
            "state": torch.arange(T * STATE_DIM, dtype=torch.float32).reshape(T, STATE_DIM),
            "action": torch.arange(T * ACTION_DIM, dtype=torch.float32).reshape(T, ACTION_DIM) + 0.5,
        }
        meta_eps[str(ep)] = {"task": task, "length": T}
    torch.save(arrays, tmp_path / ARRAYS_NAME)
    meta = {
        "version": PREPROCESS_VERSION,
        "fps": 10.0,
        "image_format": "png",
        "stats": {
            "observation.state": {"min": [0.0] * STATE_DIM, "max": [100.0] * STATE_DIM},
            "action": {"min": [0.0] * ACTION_DIM, "max": [100.0] * ACTION_DIM},
        },
        "episodes": meta_eps,
    }
    (tmp_path / META_NAME).write_text(json.dumps(meta))
    return tmp_path


class TestPreprocessedDataset:
    def test_contrato_de_chaves_e_shapes(self, fake_root):
        ds = PreprocessedLiberoDataset(fake_root, pred_horizon=3)
        assert len(ds) == 10  # 6 + 4 frames
        item = ds[0]
        assert item["observation.images.image"].shape == (3, H, W)
        assert item["observation.images.image2"].shape == (3, H, W)
        assert item["observation.state"].shape == (STATE_DIM,)
        assert item["action"].shape == (3, ACTION_DIM)
        assert item["action_is_pad"].shape == (3,)
        assert item["task"] == TASK_A
        assert item["observation.images.image"].dtype == torch.float32
        assert 0.0 <= item["observation.images.image"].min()
        assert item["observation.images.image"].max() <= 1.0

    def test_chunk_no_meio_do_episodio_sem_padding(self, fake_root):
        ds = PreprocessedLiberoDataset(fake_root, pred_horizon=3)
        item = ds[1]  # ep 0, t=1; T=6 -> chunk [1,2,3] inteiro válido
        esperado = torch.arange(6 * ACTION_DIM, dtype=torch.float32).reshape(6, ACTION_DIM) + 0.5
        assert torch.equal(item["action"], esperado[1:4])
        assert not item["action_is_pad"].any()

    def test_chunk_na_borda_repete_ultima_acao_e_marca_pad(self, fake_root):
        ds = PreprocessedLiberoDataset(fake_root, pred_horizon=3)
        item = ds[5]  # ep 0, t=5 (último frame): 1 ação válida + 2 de padding
        acoes_ep0 = torch.arange(6 * ACTION_DIM, dtype=torch.float32).reshape(6, ACTION_DIM) + 0.5
        assert torch.equal(item["action"][0], acoes_ep0[5])
        assert torch.equal(item["action"][1], acoes_ep0[5])  # repete a última
        assert torch.equal(item["action"][2], acoes_ep0[5])
        assert item["action_is_pad"].tolist() == [False, True, True]

    def test_filtros_por_task_e_por_episodio(self, fake_root):
        so_a = PreprocessedLiberoDataset(fake_root, 3, task_texts={TASK_A})
        assert so_a.episode_ids == [0] and len(so_a) == 6
        so_ep1 = PreprocessedLiberoDataset(fake_root, 3, episodes=[1])
        assert so_ep1.episode_ids == [1] and len(so_ep1) == 4
        assert so_ep1.episode_task_labels == {1: TASK_B}

    def test_ponta_a_ponta_com_o_bridge(self, fake_root):
        """DataLoader (collate padrão) -> LiberoActBridge: o contrato inteiro."""
        ds = PreprocessedLiberoDataset(fake_root, pred_horizon=3)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        state_norm, action_norm = build_normalizers_from_meta(
            ds.meta, torch.device("cpu")
        )
        bridge = LiberoActBridge(state_norm, action_norm)
        batch = next(iter(loader))
        images, state, actions, is_pad, task_texts = bridge(batch, torch.device("cpu"))
        assert images.shape == (4, 2, 3, H, W)   # (B, n_cameras, C, H, W)
        assert state.shape == (4, STATE_DIM)
        assert actions.shape == (4, 3, ACTION_DIM)
        assert is_pad.shape == (4, 3) and is_pad.dtype == torch.bool
        assert list(task_texts) == [TASK_A] * 4
        # normalização min-max [0,100] -> [-1,1] aplicada pelo bridge
        assert actions.min() >= -1.0 - 1e-6 and actions.max() <= 1.0 + 1e-6

    def test_normalizadores_dos_stats_salvos(self, fake_root):
        ds = PreprocessedLiberoDataset(fake_root, pred_horizon=3)
        state_norm, _ = build_normalizers_from_meta(ds.meta, torch.device("cpu"))
        meio = torch.full((1, STATE_DIM), 50.0)
        assert torch.allclose(state_norm.normalize(meio), torch.zeros(1, STATE_DIM))


class TestGradientAccumulation:
    def _make_model(self):
        from act_lang.models.act import ACT
        from act_lang.models.backbone import freeze_batchnorm
        torch.manual_seed(0)
        # dropout=0: sem estocasticidade dependente do shape do batch.
        # freeze_batchnorm: BN em modo train usa estatísticas DO BATCH --
        # batch 2 e batch 4 dariam forwards genuinamente diferentes e a
        # equivalência da accumulation não valeria (ressalva clássica).
        # Todos os configs reais deste projeto usam freeze_bn=True, então o
        # teste espelha o uso real; ver docstring de train_one_epoch.
        model = ACT(action_dim=ACTION_DIM, state_dim=STATE_DIM, d_model=32,
                    latent_dim=8, chunk_size=3, n_cameras=2,
                    n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
                    dropout=0.0, pretrained_backbone=False)
        freeze_batchnorm(model.vision_backbone)
        return model

    @staticmethod
    def _bridge(batch, device):
        return (batch["images"], batch["state"], batch["actions"],
                batch["is_pad"], None)

    def test_accum_2_equivale_a_batch_dobrado(self):
        """2 microbatches de 2 com accum=2 == 1 batch de 4 (mesmos pesos após
        o step). Vale porque as losses são médias e os batches têm o mesmo
        tamanho e o mesmo nº de elementos válidos: (g1+g2)/2 == g_concat."""
        from act_lang.training.loop import train_one_epoch

        torch.manual_seed(1)
        b_full = {
            "images": torch.rand(4, 2, 3, H, W),
            "state": torch.rand(4, STATE_DIM),
            "actions": torch.rand(4, 3, ACTION_DIM),
            "is_pad": torch.tensor([[False, False, True]] * 4),
        }
        halves = [{k: v[:2] for k, v in b_full.items()},
                  {k: v[2:] for k, v in b_full.items()}]

        device = torch.device("cpu")
        scaler = torch.amp.GradScaler(enabled=False)

        model_a = self._make_model()
        model_b = copy.deepcopy(model_a)  # pesos idênticos, garantido
        # O CVAE amostra z no forward de treino, e o ruído tem SHAPE diferente
        # nos dois regimes (2+2 vs 4) -- equivalência exata só existe com a
        # parte determinística. Forçamos z = mu (sample_posterior=False) nos
        # dois modelos: o que este teste isola é a matemática da accumulation,
        # não a variância do reparametrization trick.
        for m in (model_a, model_b):
            orig = m.forward
            m.forward = (lambda f: lambda *a, **k: f(
                *a, **{**k, "sample_posterior": False}
            ))(orig)
        opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-3)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-3)

        train_one_epoch(model_a, halves, self._bridge, opt_a, scaler, device,
                        kl_weight=0.0, free_bits=0.0, accum_steps=2)
        train_one_epoch(model_b, [b_full], self._bridge, opt_b, scaler, device,
                        kl_weight=0.0, free_bits=0.0, accum_steps=1)

        for (na, pa), (nb, pb) in zip(model_a.named_parameters(),
                                      model_b.named_parameters()):
            assert na == nb
            assert torch.allclose(pa, pb, atol=1e-5), (
                f"{na}: divergência máx {(pa - pb).abs().max().item():.2e}"
            )

    def test_sobra_de_microbatch_ainda_aplica_step(self):
        """3 microbatches com accum=2: o 3º não fecha um grupo completo --
        o flush final precisa aplicá-lo, senão gradiente é jogado fora."""
        from act_lang.training.loop import train_one_epoch

        torch.manual_seed(2)
        def mb():
            return {
                "images": torch.rand(2, 2, 3, H, W),
                "state": torch.rand(2, STATE_DIM),
                "actions": torch.rand(2, 3, ACTION_DIM),
                "is_pad": torch.tensor([[False, False, True]] * 2),
            }

        model = self._make_model()
        antes = copy.deepcopy(model.state_dict())
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        scaler = torch.amp.GradScaler(enabled=False)
        train_one_epoch(model, [mb(), mb(), mb()], self._bridge, opt, scaler,
                        torch.device("cpu"), kl_weight=0.0, free_bits=0.0,
                        accum_steps=2)
        mudou = any(not torch.equal(antes[k], v)
                    for k, v in model.state_dict().items()
                    if v.dtype.is_floating_point)
        assert mudou
        # e nenhum gradiente residual pendente após o flush
        assert all(p.grad is None or p.grad.abs().sum() == 0
                   for p in model.parameters() if p.requires_grad)
