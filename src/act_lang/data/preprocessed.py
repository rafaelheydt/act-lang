"""Dataset de frames PRÉ-PROCESSADOS -- o antídoto do gargalo de decodificação.

Diagnóstico que motivou isto (set/2026): época de HORAS na T4 do Colab com a
GPU ociosa -- cada __getitem__ do LeRobotDataset decodifica vídeo das 2
câmeras, por amostra, e com num_workers=0 a GPU espera a CPU decodificar
frame a frame. A decodificação é o mesmo trabalho repetido a cada época.

Solução: pagar a decodificação UMA vez (scripts/preprocess_dataset.py salva
os frames em JPEG/PNG no disco) e treinar lendo frames prontos -- leitura de
JPEG pequeno é ~ordens de magnitude mais barata que seek+decode de vídeo, e
paraleliza bem com num_workers>0.

CONTRATO: cada item replica EXATAMENTE as chaves que o LiberoActBridge
consome do pipeline lerobot+delta_timestamps (ver data/libero.py):
  - "observation.images.image"  : (C, H, W) float32 em [0, 1]
  - "observation.images.image2" : (C, H, W) float32 em [0, 1]
  - "observation.state"         : (8,) float32 CRU (bridge normaliza)
  - "action"                    : (pred_horizon, 7) float32 CRU
  - "action_is_pad"             : (pred_horizon,) bool
  - "task"                      : str (collate padrão -> lista de strings)
O bridge, o modelo e o resto do treino ficam intactos: só a origem dos
bytes muda. Suporta apenas obs_horizon=1 -- a mesma restrição já assertada
em make_delta_timestamps.

Layout em disco (escrito pelo scripts/preprocess_dataset.py):
  root/
    meta.json    -- fps, formato, stats globais (min/max de state/action,
                    copiados de LeRobotDatasetMetadata.stats: normalização
                    IDÊNTICA à dos treinos feitos no caminho lerobot),
                    e {episódio: {task, length}}
    arrays.pt    -- {ep: {"state": (T,8) f32, "action": (T,7) f32}} (RAM: ~MB)
    frames/ep{ep:05d}/c0_{t:05d}.<ext>, c1_{t:05d}.<ext>
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .normalize import MinMaxNormalizer

META_NAME = "meta.json"
ARRAYS_NAME = "arrays.pt"
FRAMES_DIR = "frames"
PREPROCESS_VERSION = 1


def load_meta(root: Path) -> dict:
    meta = json.loads((Path(root) / META_NAME).read_text())
    assert meta.get("version") == PREPROCESS_VERSION, (
        f"versão do pré-processamento ({meta.get('version')}) != esperada "
        f"({PREPROCESS_VERSION}) -- re-rode scripts/preprocess_dataset.py"
    )
    return meta


def build_normalizers_from_meta(meta: dict, device: torch.device):
    """(state_norm, action_norm) a partir dos stats salvos no meta.json.

    Os valores foram copiados de LeRobotDatasetMetadata.stats na hora do
    pré-processamento -- MESMOS min/max globais do caminho lerobot, então um
    checkpoint treinado num caminho continua válido no outro.
    """
    stats = {
        k: {"min": np.asarray(v["min"], dtype=np.float32),
            "max": np.asarray(v["max"], dtype=np.float32)}
        for k, v in meta["stats"].items()
    }
    state_norm = MinMaxNormalizer.from_lerobot_stats(stats, "observation.state").to(device)
    action_norm = MinMaxNormalizer.from_lerobot_stats(stats, "action").to(device)
    return state_norm, action_norm


class PreprocessedLiberoDataset(torch.utils.data.Dataset):
    """Lê frames pré-processados e monta chunks de ação on-the-fly.

    A montagem do chunk é aritmética barata (fatia de array em RAM); o único
    I/O por amostra são 2 JPEGs pequenos -- é isso que destrava a GPU.

    Padding do chunk: além do fim do episódio, repete a última ação válida e
    marca action_is_pad=True -- mesmo comportamento de borda do lerobot; o
    valor em si é indiferente porque a loss mascara essas posições.
    """

    def __init__(
        self,
        root: str | Path,
        pred_horizon: int,
        task_texts: set[str] | list[str] | None = None,
        episodes: list[int] | None = None,
    ):
        self.root = Path(root)
        self.pred_horizon = pred_horizon
        self.meta = load_meta(self.root)
        self._ext = self.meta.get("image_format", "jpg")

        all_eps = {int(k): v for k, v in self.meta["episodes"].items()}
        selected = set(all_eps)
        if task_texts is not None:
            wanted = set(task_texts)
            selected &= {ep for ep, info in all_eps.items() if info["task"] in wanted}
        if episodes is not None:
            selected &= set(int(e) for e in episodes)
        self.episode_ids = sorted(selected)
        assert self.episode_ids, "nenhum episódio após filtros (task_texts/episodes)"

        arrays = torch.load(self.root / ARRAYS_NAME, map_location="cpu",
                            weights_only=False)
        self._state = {ep: arrays[ep]["state"].float() for ep in self.episode_ids}
        self._action = {ep: arrays[ep]["action"].float() for ep in self.episode_ids}

        # índice plano: amostra i -> (episódio, t); todo frame vira uma amostra,
        # como no caminho lerobot.
        self._index: list[tuple[int, int]] = []
        for ep in self.episode_ids:
            length = all_eps[ep]["length"]
            assert length == self._state[ep].shape[0], (
                f"ep {ep}: length do meta ({length}) != arrays "
                f"({self._state[ep].shape[0]}) -- pré-processamento incompleto?"
            )
            self._index.extend((ep, t) for t in range(length))
        self._task = {ep: all_eps[ep]["task"] for ep in self.episode_ids}

    @property
    def episode_task_labels(self) -> dict[int, str]:
        """{episódio: task string} -- mesmo formato que o caminho lerobot usa
        para os splits (split_episodes_min_holdout / stratified)."""
        return dict(self._task)

    def _load_frame(self, ep: int, cam: int, t: int) -> torch.Tensor:
        path = self.root / FRAMES_DIR / f"ep{ep:05d}" / f"c{cam}_{t:05d}.{self._ext}"
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        ep, t = self._index[i]
        actions = self._action[ep]
        length = actions.shape[0]
        end = min(t + self.pred_horizon, length)
        chunk = actions[t:end]
        n_pad = self.pred_horizon - chunk.shape[0]
        if n_pad > 0:
            chunk = torch.cat([chunk, chunk[-1:].expand(n_pad, -1)], dim=0)
        is_pad = torch.zeros(self.pred_horizon, dtype=torch.bool)
        if n_pad > 0:
            is_pad[-n_pad:] = True

        return {
            "observation.images.image": self._load_frame(ep, 0, t),
            "observation.images.image2": self._load_frame(ep, 1, t),
            "observation.state": self._state[ep][t],
            "action": chunk,
            "action_is_pad": is_pad,
            "task": self._task[ep],
        }
