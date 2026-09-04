"""Dataset LIBERO lendo HDF5 nativo (arrays uint8 crus, sem decodificação de
imagem) -- segunda etapa da otimização de I/O, depois do pré-processamento
pra JPEG (ver data/preprocessed.py). Ver docs/hdf5_migration.md pro
raciocínio completo: benchmarks de I/O, mapeamento das 40 tarefas pros
arquivos oficiais e a verificação de convenção de ação/imagem que molda
esta implementação.

CONTRATO: mesmas chaves de PreprocessedLiberoDataset -- o LiberoActBridge,
o modelo e o resto do treino ficam intactos, só a origem dos bytes muda:
  - "observation.images.image"  : (C, 128, 128) float32 em [0, 1]
  - "observation.images.image2" : (C, 128, 128) float32 em [0, 1]
  - "observation.state"         : (8,) float32 CRU (bridge normaliza)
  - "action"                    : (pred_horizon, 7) float32 CRU
  - "action_is_pad"             : (pred_horizon,) bool
  - "task"                      : str

Achados que moldam esta implementação (docs/hdf5_migration.md tem o
detalhe e a evidência):
  - Convenção de ação idêntica à do pipeline lerobot (OSC_POSE: delta xyz +
    delta eixo-ângulo + gripper) -- sem conversão de espaço de ação.
  - obs/ee_states (6,) == concat(obs/ee_pos, obs/ee_ori); concatenado com
    obs/gripper_states (2,) reconstitui observation.state (8,) sem
    quat2axisangle -- o HDF5 já guarda ee_ori em eixo-ângulo (quat2axisangle
    só é necessário no caminho do env AO VIVO, ver eval/obs_processing.py,
    que converte um quaternion cru do simulador).
  - IMAGENS VÊM INVERTIDAS VERTICALMENTE em relação à convenção do vídeo
    lerobot (confirmado por comparação pixel-a-pixel entre um frame do
    HDF5 oficial e o frame correspondente do vídeo lerobot/libero -- sem o
    flip, o braço do robô aparece "pendurado no teto"). Corrigido em
    `_frame_to_tensor` com `arr[::-1]` (flip no eixo H) -- sem isso o
    treino aprenderia de imagens de cabeça para baixo, silenciosamente.

Handles de arquivo HDF5 não são picklable/seguros através de fork: abertos
sob demanda em __getitem__ e cacheados por (path, os.getpid()) -- cada
worker process abre uma vez e reusa (reabrir custa ~15-250ms dependendo da
profundidade da árvore de grupos internos, ver benchmark em
docs/hdf5_migration.md). __init__ roda no processo PRINCIPAL, antes do fork
dos workers do DataLoader: só abre cada arquivo pra ler metadados (nomes e
comprimento dos demos) e fecha em seguida com `with` -- nunca guarda um
handle que sobreviveria ao fork.
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .normalize import MinMaxNormalizer

META_NAME = "meta.json"
TASK_MAPPING_NAME = "task_mapping.json"
HDF5_VERSION = 1


def load_task_mapping(root: str | Path) -> dict[str, tuple[str, str]]:
    """{task_text: (suite, filename)} a partir de root/task_mapping.json
    (cópia de configs/task_mapping_libero40.json, feita por
    scripts/download_libero_hdf5.py na hora do download -- mesmo padrão de
    root/meta.json em data/preprocessed.py: o dado carrega seus próprios
    metadados, o código não hardcoda caminho pra dentro de configs/)."""
    data = json.loads((Path(root) / TASK_MAPPING_NAME).read_text(encoding="utf-8"))
    return {task: (v["suite"], v["file"]) for task, v in data["tasks"].items()}


def load_meta(root: str | Path) -> dict:
    meta = json.loads((Path(root) / META_NAME).read_text())
    assert meta.get("version") == HDF5_VERSION, (
        f"versão do meta.json HDF5 ({meta.get('version')}) != esperada "
        f"({HDF5_VERSION}) -- re-rode scripts/download_libero_hdf5.py"
    )
    return meta


def build_normalizers_from_meta(meta: dict, device: torch.device):
    """(state_norm, action_norm) a partir dos stats salvos no meta.json --
    calculados sobre os próprios arquivos HDF5 (scripts/download_libero_hdf5.py),
    não copiados do lerobot: mesmo a convenção sendo idêntica (ver
    docs/hdf5_migration.md), o subconjunto de episódios por tarefa difere
    (lerobot/libero usa só uma fração dos demos oficiais por tarefa), então
    min/max exatos não são garantidamente os mesmos."""
    stats = {
        k: {"min": np.asarray(v["min"], dtype=np.float32),
            "max": np.asarray(v["max"], dtype=np.float32)}
        for k, v in meta["stats"].items()
    }
    state_norm = MinMaxNormalizer.from_lerobot_stats(stats, "observation.state").to(device)
    action_norm = MinMaxNormalizer.from_lerobot_stats(stats, "action").to(device)
    return state_norm, action_norm


def compute_stats(root: str | Path, task_mapping: dict | None = None) -> dict:
    """Escaneia todos os arquivos HDF5 mapeados e computa min/max globais de
    `action` e `observation.state` -- vira o "stats" do meta.json. Lê os
    arquivos inteiros (só as datasets pequenas de state/action, não as
    imagens) -- rodar uma vez por download, não por treino."""
    root = Path(root)
    mapping = task_mapping if task_mapping is not None else load_task_mapping(root)
    action_min = action_max = state_min = state_max = None
    for task, (suite, fname) in sorted(mapping.items()):
        path = root / suite / fname
        with h5py.File(path, "r") as f:
            for demo_key in f["data"].keys():
                demo = f["data"][demo_key]
                actions = demo["actions"][:].astype(np.float32)
                state = np.concatenate(
                    [demo["obs"]["ee_states"][:], demo["obs"]["gripper_states"][:]],
                    axis=1,
                ).astype(np.float32)
                a_min, a_max = actions.min(axis=0), actions.max(axis=0)
                s_min, s_max = state.min(axis=0), state.max(axis=0)
                action_min = a_min if action_min is None else np.minimum(action_min, a_min)
                action_max = a_max if action_max is None else np.maximum(action_max, a_max)
                state_min = s_min if state_min is None else np.minimum(state_min, s_min)
                state_max = s_max if state_max is None else np.maximum(state_max, s_max)
    return {
        "action": {"min": action_min.tolist(), "max": action_max.tolist()},
        "observation.state": {"min": state_min.tolist(), "max": state_max.tolist()},
    }


class HDF5LiberoDataset(torch.utils.data.Dataset):
    """Lê episódios direto do HDF5 oficial do LIBERO e monta chunks de ação
    on-the-fly -- mesma aritmética de padding de PreprocessedLiberoDataset.

    `episodes`, quando passado, usa os IDs inteiros globais expostos por
    `episode_task_labels` (mesmo padrão de PreprocessedLiberoDataset) --
    não são os `demo_i` do HDF5 diretamente, porque um único episódio no
    sentido deste dataset é (tarefa, demo) e precisa de um ID plano único
    pra alimentar split_episodes_min_holdout/stratified sem mudar essas
    funções.
    """

    def __init__(
        self,
        root: str | Path,
        pred_horizon: int,
        task_texts: set[str] | list[str] | None = None,
        episodes: list[int] | None = None,
        task_mapping: dict[str, tuple[str, str]] | None = None,
    ):
        self.root = Path(root)
        self.pred_horizon = pred_horizon
        mapping = task_mapping if task_mapping is not None else load_task_mapping(self.root)
        if task_texts is not None:
            wanted = set(task_texts)
            mapping = {t: v for t, v in mapping.items() if t in wanted}
        assert mapping, "nenhuma tarefa após filtro de task_texts"

        # Metadados (comprimento de cada demo) lidos UMA vez no processo
        # principal; o `with` fecha o handle antes do DataLoader dar fork
        # nos workers -- ver docstring do módulo.
        self._episode_meta: dict[int, dict] = {}
        eid = 0
        for task, (suite, fname) in sorted(mapping.items()):
            path = self.root / suite / fname
            with h5py.File(path, "r") as f:
                demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
                for demo_key in demo_keys:
                    length = f["data"][demo_key]["actions"].shape[0]
                    self._episode_meta[eid] = {
                        "task": task, "path": path, "demo_key": demo_key, "length": length,
                    }
                    eid += 1

        selected = set(self._episode_meta)
        if episodes is not None:
            selected &= set(int(e) for e in episodes)
        self.episode_ids = sorted(selected)
        assert self.episode_ids, "nenhum episódio após filtros (task_texts/episodes)"

        self._index: list[tuple[int, int]] = []
        for eid in self.episode_ids:
            length = self._episode_meta[eid]["length"]
            self._index.extend((eid, t) for t in range(length))

        # cache de handle por (path, pid) -- ver docstring do módulo.
        self._file_cache: dict[Path, tuple[int, h5py.File]] = {}

    @property
    def episode_task_labels(self) -> dict[int, str]:
        """{episódio: task string} -- mesmo formato que PreprocessedLiberoDataset,
        consumido por split_episodes_min_holdout/stratified sem alteração."""
        return {eid: self._episode_meta[eid]["task"] for eid in self.episode_ids}

    def _get_file(self, path: Path) -> h5py.File:
        pid = os.getpid()
        cached = self._file_cache.get(path)
        if cached is not None and cached[0] == pid:
            return cached[1]
        f = h5py.File(path, "r")
        self._file_cache[path] = (pid, f)
        return f

    def _demo(self, eid: int):
        meta = self._episode_meta[eid]
        return self._get_file(meta["path"])["data"][meta["demo_key"]]

    @staticmethod
    def _frame_to_tensor(arr: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(arr[::-1])  # flip vertical -- ver docstring do módulo
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        eid, t = self._index[i]
        demo = self._demo(eid)
        obs = demo["obs"]

        actions_ds = demo["actions"]
        length = actions_ds.shape[0]
        end = min(t + self.pred_horizon, length)
        chunk = torch.from_numpy(actions_ds[t:end].astype(np.float32))
        n_pad = self.pred_horizon - chunk.shape[0]
        if n_pad > 0:
            chunk = torch.cat([chunk, chunk[-1:].expand(n_pad, -1)], dim=0)
        is_pad = torch.zeros(self.pred_horizon, dtype=torch.bool)
        if n_pad > 0:
            is_pad[-n_pad:] = True

        state = np.concatenate(
            [obs["ee_states"][t], obs["gripper_states"][t]]
        ).astype(np.float32)

        return {
            "observation.images.image": self._frame_to_tensor(obs["agentview_rgb"][t]),
            "observation.images.image2": self._frame_to_tensor(obs["eye_in_hand_rgb"][t]),
            "observation.state": torch.from_numpy(state),
            "action": chunk,
            "action_is_pad": is_pad,
            "task": self._episode_meta[eid]["task"],
        }
