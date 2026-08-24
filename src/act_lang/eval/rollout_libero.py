"""Rollout no LiberoEnv com temporal ensembling (ACT, Zhao et al. 2023).

CORREÇÃO aplicada: os pesos exp(-m*i) seguem a convenção do paper — w0 é a
predição MAIS ANTIGA (favorece suavidade). O notebook original invertia a
ordem, dando peso máximo à predição mais recente.
"""

from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch

from ..data.normalize import MinMaxNormalizer
from .obs_processing import process_libero_obs


def temporal_ensemble(action_buffer: deque, m: float) -> np.ndarray:
    """Combina as predições da ação ATUAL vindas dos últimos L chunks.

    buffer[i] foi predito há (L-1-i) passos, logo a ação atual está no índice
    (L-1-i) desse chunk. Ordenamos do mais antigo ao mais novo e aplicamos
    w_i = exp(-m*i) com w0 = mais antigo (convenção do paper).
    """
    L = len(action_buffer)
    preds = np.stack([action_buffer[i][L - 1 - i] for i in range(L)])  # antigo -> novo
    weights = np.exp(-m * np.arange(L))
    weights /= weights.sum()
    return weights @ preds


@torch.no_grad()
def rollout_libero(
    model,
    env,
    state_norm: MinMaxNormalizer,
    action_norm: MinMaxNormalizer,
    device: torch.device,
    n_episodes: int = 10,
    m: float = 0.01,
    max_steps: int = 300,
    video_dir: str | Path = "libero_rollouts",
    seed_start: int = 1000,
    video_fps: int = 20,
    task_text: str | None = None,  # Fase 2: passa a instrução pro modelo
) -> list[dict]:
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    results = []
    for ep in range(n_episodes):
        raw_obs, info = env.reset(seed=seed_start + ep)
        frames = [np.concatenate(
            [raw_obs["pixels"]["image"], raw_obs["pixels"]["image2"]], axis=1
        )]
        action_buffer: deque = deque(maxlen=model.chunk_size)

        for step in range(max_steps):
            images, state = process_libero_obs(raw_obs, state_norm, device)
            task_texts = [task_text] if task_text is not None else None
            pred_actions, _, _ = model(
                images, state, actions=None, task_texts=task_texts
            )  # z = 0
            pred_actions = action_norm.denormalize(pred_actions)
            action_buffer.append(pred_actions[0].cpu().numpy())

            action = temporal_ensemble(action_buffer, m)
            raw_obs, reward, terminated, truncated, info = env.step(
                action.astype(np.float32)
            )
            frames.append(np.concatenate(
                [raw_obs["pixels"]["image"], raw_obs["pixels"]["image2"]], axis=1
            ))
            if terminated or truncated:
                break

        success = bool(info.get("is_success", False))
        results.append({"success": success, "steps": step + 1})
        tag = "sucesso" if success else "falha"
        imageio.mimsave(video_dir / f"ep{ep:02d}_{tag}.mp4", frames, fps=video_fps)
        print(f"episódio {ep + 1}/{n_episodes}: sucesso={success} | steps={step + 1}")

    return results
