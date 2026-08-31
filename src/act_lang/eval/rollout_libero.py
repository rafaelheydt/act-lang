"""Rollout no LiberoEnv com temporal ensembling (ACT, Zhao et al. 2023).

CORREÇÕES aplicadas:
  1. Pesos exp(-m*i) seguem a convenção do paper — w0 é a predição MAIS
     ANTIGA (favorece suavidade). O notebook original invertia a ordem.
  2. Seleção de cenário via `init_state_id`, não via `seed`. Conferido no
     código-fonte de lerobot/envs/libero.py: `reset(seed=...)` usa `seed`
     só pra `self._env.seed(seed)` (ruído de física/render) -- quem escolhe
     o layout inicial é `self.init_state_id`, indexando um array de estados
     pré-computados (`self._init_states`), e esse índice AUTO-INCREMENTA a
     cada `reset()`, dependendo do histórico de chamadas anteriores no mesmo
     objeto `env`. Confiar nisso implicitamente é frágil (qualquer reset
     de diagnóstico feito antes bagunça a sequência); por isso, aqui,
     `init_state_id` é setado explicitamente a cada episódio.
  3. `init_state_ids` (lista explícita, opcional) permite amostrar cenários
     ESPALHADOS pelo intervalo inteiro (ex: [0,5,10,...,45]), não só um
     bloco consecutivo -- útil pra verificar que o sucesso não é artefato
     de testar sempre a mesma vizinhança de índices baixos.
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


def _video_frame(raw_obs: dict) -> np.ndarray:
    """Frame lado a lado das 2 câmeras, com o MESMO flip de 180° aplicado à
    entrada do modelo (ver obs_processing) — os pixels crus do env vêm
    invertidos em relação à convenção do dataset; sem o flip, os vídeos de
    inspeção de falhas (a razão de eles existirem) saíam de cabeça pra baixo.
    """
    frame = np.concatenate(
        [raw_obs["pixels"]["image"], raw_obs["pixels"]["image2"]], axis=1
    )
    return frame[::-1, ::-1]  # 180°: inverte H e W, canais intactos


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
    init_state_start: int = 0,  # usado só se init_state_ids=None
    init_state_ids: list[int] | None = None,  # CORREÇÃO 3: amostragem espalhada
) -> list[dict]:
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    n_init_states = len(env._init_states) if getattr(env, "_init_states", None) is not None else None

    if init_state_ids is None:
        init_state_ids = [init_state_start + ep for ep in range(n_episodes)]
    n_episodes = len(init_state_ids)

    if n_init_states is not None and max(init_state_ids) >= n_init_states:
        print(
            f"aviso: init_state_id máximo pedido ({max(init_state_ids)}) >= "
            f"{n_init_states} init_states disponíveis -- vai dar módulo "
            f"(cenários repetidos), conforme o comportamento do próprio env."
        )

    results = []
    for ep, state_id in enumerate(init_state_ids):
        # CORREÇÃO 2: init_state_id escolhe o cenário; seed só afeta ruído
        # de física/render (não reposiciona objetos).
        env.init_state_id = state_id
        raw_obs, info = env.reset(seed=seed_start + ep)
        frames = [_video_frame(raw_obs)]
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
            frames.append(_video_frame(raw_obs))
            if terminated or truncated:
                break

        success = bool(info.get("is_success", False))
        results.append({"success": success, "steps": step + 1, "init_state_id": state_id})
        tag = "sucesso" if success else "falha"
        imageio.mimsave(video_dir / f"ep{ep:02d}_state{state_id:02d}_{tag}.mp4", frames, fps=video_fps)
        print(
            f"episódio {ep + 1}/{n_episodes} (init_state_id={state_id}): "
            f"sucesso={success} | steps={step + 1}"
        )

    return results
