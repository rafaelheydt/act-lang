"""Pipeline de dados do LIBERO via lerobot.

Cobre: filtro de episódios por tarefa (sem decodificar vídeo), split por
episódio, delta_timestamps e a ponte batch -> entradas do ACT.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .normalize import MinMaxNormalizer

REPO_ID = "lerobot/libero"


def filter_episodes_by_tasks(meta, full_dataset, task_texts: set[str]) -> list[int]:
    """Episódios cujo texto de tarefa está em `task_texts`.

    A tarefa só existe por FRAME (não em meta.episodes); usamos select_columns
    p/ ler apenas task_index (sem decodificar vídeo) e consultamos o primeiro
    frame de cada episódio — a tarefa é constante dentro do episódio.
    """
    task_index_col = full_dataset.select_columns(["task_index"])
    task_index_to_text = dict(zip(meta.tasks["task_index"], meta.tasks.index))

    episode_ids = []
    for ep in meta.episodes:
        first_frame_idx = ep["dataset_from_index"]
        t_idx = int(task_index_col[first_frame_idx]["task_index"])
        if task_index_to_text[t_idx] in task_texts:
            episode_ids.append(ep["episode_index"])
    return episode_ids


def split_episodes(
    episode_ids: list[int], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split por EPISÓDIO (nunca por frame — frames do mesmo episódio em
    train e val inflariam a validação por quase-duplicatas)."""
    ids = np.array(episode_ids)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(ids)
    n_val = max(1, int(val_frac * len(ids)))
    return ids[n_val:].tolist(), ids[:n_val].tolist()


def make_delta_timestamps(fps: float, obs_horizon: int = 1, pred_horizon: int = 50) -> dict:
    return {
        "observation.images.image": [-i / fps for i in reversed(range(obs_horizon))],
        "observation.images.image2": [-i / fps for i in reversed(range(obs_horizon))],
        "observation.state": [-i / fps for i in reversed(range(obs_horizon))],
        "action": [i / fps for i in range(pred_horizon)],
    }


@dataclass
class LiberoActBridge:
    """Converte um batch do LeRobotDataset nas entradas do ACT, já normalizadas.

    Observado empiricamente (smoke test): com obs_horizon=1, as imagens vêm SEM
    a dimensão temporal; o estado vem COM ela (por isso o squeeze(1)).
    """

    state_norm: MinMaxNormalizer
    action_norm: MinMaxNormalizer

    def __call__(self, batch: dict, device: torch.device):
        images = torch.stack(
            [batch["observation.images.image"], batch["observation.images.image2"]],
            dim=1,
        ).to(device)  # (B, n_cameras=2, C, H, W), em [0,1] — ImageNet norm é do modelo
        state = self.state_norm.normalize(
            batch["observation.state"].squeeze(1).to(device)
        )  # (B, state_dim)
        actions = self.action_norm.normalize(batch["action"].to(device))
        is_pad = batch["action_is_pad"].to(device)
        task_texts = batch["task"]  # lista de strings; usada na Fase 2
        return images, state, actions, is_pad, task_texts
