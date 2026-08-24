"""Ponte entre a observação CRUA do LiberoEnv e as entradas do ACT.

As duas conversões abaixo foram descobertas lendo o código-fonte do lerobot
(lerobot/processor/env_processor.py, LiberoProcessorStep):
  1. O vetor de 8 dims do dataset é eef_pos(3) + quat2axisangle(eef_quat)(3)
     + gripper_qpos(2) — o env devolve os componentes crus, separados.
  2. As imagens do env vêm invertidas 180° em relação ao dataset
     (torch.flip nos eixos H e W — convenção de câmera do LIBERO).

quat2axisangle tem casos de teste em tests/test_obs_processing.py — se o
lerobot mudar a convenção um dia, os testes acusam.
"""

import numpy as np
import torch

from ..data.normalize import MinMaxNormalizer


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """quat (4,) em formato (x, y, z, w) -> axis-angle (3,).

    Mesma fórmula do LiberoProcessorStep oficial do lerobot.
    """
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = quat[:3] / den
    return axis * angle


def process_libero_obs(
    raw_obs: dict, state_norm: MinMaxNormalizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduz a transformação do LiberoProcessorStep.

    Imagens saem em [0, 1] SEM normalização ImageNet — ela vive dentro do
    VisionBackbone e viaja com o checkpoint (ver models/backbone.py).
    """
    img1 = raw_obs["pixels"]["image"]
    img2 = raw_obs["pixels"]["image2"]
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    img1_t = torch.flip(img1_t, dims=[1, 2])  # 180° — convenção do LIBERO
    img2_t = torch.flip(img2_t, dims=[1, 2])
    images = torch.stack([img1_t, img2_t], dim=0).unsqueeze(0).to(device)  # (1,2,C,H,W)

    eef_pos = raw_obs["robot_state"]["eef"]["pos"]
    eef_quat = raw_obs["robot_state"]["eef"]["quat"]
    gripper_qpos = raw_obs["robot_state"]["gripper"]["qpos"]
    state_np = np.concatenate([eef_pos, quat2axisangle(eef_quat), gripper_qpos])
    state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)  # (1, 8)
    return images, state_norm.normalize(state)
