"""Testes do caminho de dados HDF5 nativo (data/hdf5_libero.py).

O que importa garantir aqui:
1. HDF5LiberoDataset honra o MESMO contrato de batch que
   PreprocessedLiberoDataset (chaves, shapes, padding do chunk) -- é isso
   que deixa o LiberoActBridge e o resto do treino intactos.
2. O flip vertical de imagem é aplicado (ver docstring do módulo -- achado
   de que o HDF5 oficial guarda frames invertidos em relação à convenção
   do vídeo lerobot).
3. observation.state é reconstituído como concat(ee_states, gripper_states).
4. Handles de arquivo são cacheados por processo, não reabertos por amostra.
"""

import json

import h5py
import numpy as np
import pytest
import torch

from act_lang.data.hdf5_libero import (
    HDF5_VERSION, META_NAME, TASK_MAPPING_NAME,
    HDF5LiberoDataset, build_normalizers_from_meta, compute_stats, load_task_mapping,
)
from act_lang.data.libero import LiberoActBridge

H, W = 8, 8
STATE_DIM, ACTION_DIM = 8, 7
TASK_A = "pick up the milk and place it in the basket"
TASK_B = "pick up the ketchup and place it in the basket"


def _write_demo_file(path, task_lengths: dict[str, int], rng):
    """Escreve um HDF5 no layout oficial do LIBERO com N demos, 1 por
    entrada de task_lengths (chave = nome do demo, ignorado aqui -- 1
    arquivo = 1 tarefa, como nos arquivos oficiais)."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for demo_key, T in task_lengths.items():
            demo = data.create_group(demo_key)
            obs = demo.create_group("obs")
            # imagem crua "de cabeça para baixo": linha 0 = valor alto,
            # última linha = valor baixo -- flip vertical deve inverter isso.
            agentview = np.zeros((T, H, W, 3), dtype=np.uint8)
            eye_in_hand = np.zeros((T, H, W, 3), dtype=np.uint8)
            for t in range(T):
                for row in range(H):
                    agentview[t, row] = (row * 10 + t) % 256
                    eye_in_hand[t, row] = (row * 5 + t) % 256
            obs.create_dataset("agentview_rgb", data=agentview)
            obs.create_dataset("eye_in_hand_rgb", data=eye_in_hand)
            ee_states = rng.uniform(-1, 1, size=(T, 6)).astype(np.float64)
            gripper_states = rng.uniform(-1, 1, size=(T, 2)).astype(np.float64)
            obs.create_dataset("ee_states", data=ee_states)
            obs.create_dataset("gripper_states", data=gripper_states)
            actions = rng.uniform(-1, 1, size=(T, ACTION_DIM)).astype(np.float64)
            demo.create_dataset("actions", data=actions)


@pytest.fixture()
def fake_root(tmp_path):
    """root/task_mapping.json + root/{suite}/{file}.hdf5 -- layout que
    scripts/download_libero_hdf5.py produz. 2 tarefas (1 arquivo cada), T=6
    e T=4, mimetizando o fixture equivalente de test_preprocessed.py."""
    rng = np.random.default_rng(0)
    suite = "libero_object"
    (tmp_path / suite).mkdir()

    file_a = f"{suite}/task_a_demo.hdf5"
    file_b = f"{suite}/task_b_demo.hdf5"
    _write_demo_file(tmp_path / file_a, {"demo_0": 6}, rng)
    _write_demo_file(tmp_path / file_b, {"demo_0": 4}, rng)

    mapping = {
        "tasks": {
            TASK_A: {"suite": suite, "file": "task_a_demo.hdf5"},
            TASK_B: {"suite": suite, "file": "task_b_demo.hdf5"},
        }
    }
    (tmp_path / TASK_MAPPING_NAME).write_text(json.dumps(mapping))

    stats = compute_stats(tmp_path, load_task_mapping(tmp_path))
    meta = {"version": HDF5_VERSION, "resolution": H, "stats": stats}
    (tmp_path / META_NAME).write_text(json.dumps(meta))
    return tmp_path


class TestHDF5Dataset:
    def test_contrato_de_chaves_e_shapes(self, fake_root):
        ds = HDF5LiberoDataset(fake_root, pred_horizon=3)
        assert len(ds) == 10  # 6 + 4 frames
        item = ds[0]
        assert item["observation.images.image"].shape == (3, H, W)
        assert item["observation.images.image2"].shape == (3, H, W)
        assert item["observation.state"].shape == (STATE_DIM,)
        assert item["action"].shape == (3, ACTION_DIM)
        assert item["action_is_pad"].shape == (3,)
        assert item["observation.images.image"].dtype == torch.float32
        assert item["observation.state"].dtype == torch.float32
        assert item["action"].dtype == torch.float32
        assert 0.0 <= item["observation.images.image"].min()
        assert item["observation.images.image"].max() <= 1.0

    def test_flip_vertical_aplicado(self, fake_root):
        """Linha 0 do array cru vira a ÚLTIMA linha do tensor (C,H,W)."""
        ds = HDF5LiberoDataset(fake_root, pred_horizon=3, task_texts={TASK_A})
        with h5py.File(fake_root / "libero_object" / "task_a_demo.hdf5", "r") as f:
            raw = f["data"]["demo_0"]["obs"]["agentview_rgb"][0]  # (H,W,3)
        item = ds[0]
        img = item["observation.images.image"]  # (3,H,W) float [0,1]
        raw_first_row = torch.from_numpy(raw[0].astype(np.float32) / 255.0)  # (W,3)
        # raw linha 0 -> deve aparecer na ÚLTIMA linha do tensor pós-flip
        assert torch.allclose(img[:, -1, :].permute(1, 0), raw_first_row, atol=1e-6)

    def test_state_e_concat_ee_states_gripper_states(self, fake_root):
        ds = HDF5LiberoDataset(fake_root, pred_horizon=3, task_texts={TASK_A})
        with h5py.File(fake_root / "libero_object" / "task_a_demo.hdf5", "r") as f:
            demo = f["data"]["demo_0"]
            esperado = np.concatenate(
                [demo["obs"]["ee_states"][0], demo["obs"]["gripper_states"][0]]
            ).astype(np.float32)
        item = ds[0]
        assert torch.allclose(item["observation.state"], torch.from_numpy(esperado))

    def test_chunk_na_borda_repete_ultima_acao_e_marca_pad(self, fake_root):
        ds = HDF5LiberoDataset(fake_root, pred_horizon=3, task_texts={TASK_A})
        with h5py.File(fake_root / "libero_object" / "task_a_demo.hdf5", "r") as f:
            acoes_ep0 = f["data"]["demo_0"]["actions"][:].astype(np.float32)
        item = ds[5]  # único episódio deste dataset filtrado (task_a), t=5 (último frame, T=6)
        assert torch.allclose(item["action"][0], torch.from_numpy(acoes_ep0[5]))
        assert torch.allclose(item["action"][1], torch.from_numpy(acoes_ep0[5]))
        assert torch.allclose(item["action"][2], torch.from_numpy(acoes_ep0[5]))
        assert item["action_is_pad"].tolist() == [False, True, True]

    def test_filtros_por_task_e_por_episodio(self, fake_root):
        so_a = HDF5LiberoDataset(fake_root, 3, task_texts={TASK_A})
        assert len(so_a) == 6
        assert set(so_a.episode_task_labels.values()) == {TASK_A}

        probe = HDF5LiberoDataset(fake_root, 3)
        all_ids = sorted(probe.episode_task_labels)
        assert len(all_ids) == 2  # 2 episódios (1 demo por tarefa)
        so_ep = HDF5LiberoDataset(fake_root, 3, episodes=[all_ids[1]])
        assert so_ep.episode_ids == [all_ids[1]]

    def test_handle_de_arquivo_e_reusado_no_mesmo_processo(self, fake_root):
        ds = HDF5LiberoDataset(fake_root, pred_horizon=3)
        _ = ds[0]
        _ = ds[1]
        assert len(ds._file_cache) == 1  # só 1 arquivo tocado (task_a), 1 handle cacheado
        (path, (pid, handle)), = ds._file_cache.items()
        _ = ds[2]
        (_, (_, handle2)), = ds._file_cache.items()
        assert handle is handle2  # mesmo objeto -- não reabriu

    def test_ponta_a_ponta_com_o_bridge(self, fake_root):
        ds = HDF5LiberoDataset(fake_root, pred_horizon=3)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        meta = json.loads((fake_root / META_NAME).read_text())
        state_norm, action_norm = build_normalizers_from_meta(meta, torch.device("cpu"))
        bridge = LiberoActBridge(state_norm, action_norm)
        batch = next(iter(loader))
        images, state, actions, is_pad, task_texts = bridge(batch, torch.device("cpu"))
        assert images.shape == (4, 2, 3, H, W)
        assert state.shape == (4, STATE_DIM)
        assert actions.shape == (4, 3, ACTION_DIM)
        assert is_pad.shape == (4, 3) and is_pad.dtype == torch.bool
        assert actions.min() >= -1.0 - 1e-4 and actions.max() <= 1.0 + 1e-4
