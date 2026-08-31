"""Pipeline de dados do LIBERO via lerobot.

Cobre: filtro de episódios por tarefa (sem decodificar vídeo), split por
episódio (simples e estratificado por tarefa), delta_timestamps e a ponte
batch -> entradas do ACT.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .normalize import MinMaxNormalizer

REPO_ID = "lerobot/libero"


def _select_task_index_column(full_dataset):
    """Compatibilidade entre versões do lerobot.

    Em algumas versões, LeRobotDataset expõe select_columns() diretamente.
    Na 0.4.4 (confirmado rodando local -- AttributeError na versão anterior
    deste arquivo), esse método sumiu do LeRobotDataset; o acesso passa a
    ser via o datasets.Dataset interno, em `full_dataset.hf_dataset`
    (select_columns é API da lib `datasets` da HuggingFace, não do lerobot
    em si -- por isso sobrevive lá independente da versão do lerobot).
    """
    if hasattr(full_dataset, "select_columns"):
        return full_dataset.select_columns(["task_index"])
    return full_dataset.hf_dataset.select_columns(["task_index"])


def _episode_task_texts(meta, full_dataset) -> dict[int, str]:
    """Mapa episode_id -> texto da tarefa, para TODOS os episódios do dataset.

    Lê task_index só do primeiro frame de cada episódio (via select_columns,
    sem decodificar vídeo) -- a tarefa é constante dentro do episódio. Função
    interna compartilhada por filter_episodes_by_tasks e
    get_episode_task_labels, pra manter um único lugar fazendo esse lookup.
    """
    task_index_col = _select_task_index_column(full_dataset)
    task_index_to_text = dict(zip(meta.tasks["task_index"], meta.tasks.index))
    result = {}
    for ep in meta.episodes:
        t_idx = int(task_index_col[ep["dataset_from_index"]]["task_index"])
        result[ep["episode_index"]] = task_index_to_text[t_idx]
    return result


def filter_episodes_by_tasks(meta, full_dataset, task_texts: set[str]) -> list[int]:
    """Episódios cujo texto de tarefa está em `task_texts`."""
    labels = _episode_task_texts(meta, full_dataset)
    return [eid for eid, text in labels.items() if text in task_texts]


def get_episode_task_labels(meta, full_dataset, episode_ids: list[int]) -> dict[int, str]:
    """Texto da tarefa de cada episode_id em `episode_ids` -- usado pelo split
    ESTRATIFICADO (ver split_episodes_stratified). Reaproveita o mesmo lookup
    de filter_episodes_by_tasks, sem duplicar a lógica de leitura."""
    all_labels = _episode_task_texts(meta, full_dataset)
    return {eid: all_labels[eid] for eid in episode_ids}


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


def split_episodes_stratified(
    episode_ids: list[int],
    episode_task_labels: dict[int, str],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Como split_episodes, mas embaralha e corta INDEPENDENTEMENTE dentro de
    cada tarefa, depois junta os resultados -- garante que toda tarefa com
    >=2 episódios apareça em train E val.

    Importante com poucos episódios por tarefa (ex: Fase 2, ~5/tarefa): um
    split GLOBAL corre risco real de deixar alguma tarefa inteiramente de um
    lado só por sorte do embaralhamento, o que inviabilizaria comparar a
    métrica de validação entre tarefas.

    Com 1 única tarefa em `episode_task_labels`, degenera exatamente pro
    mesmo resultado de split_episodes (um grupo só) -- seguro usar sempre,
    inclusive na Fase 1.

    Tarefas com exatamente 1 episódio ficam inteiras em train (não há como
    dividir 1 episódio); um aviso é impresso quando isso acontece.
    """
    by_task: dict[str, list[int]] = defaultdict(list)
    for eid in episode_ids:
        by_task[episode_task_labels[eid]].append(eid)

    train_ids, val_ids = [], []
    for task_text, ids_for_task in by_task.items():
        if len(ids_for_task) < 2:
            print(
                f"aviso: tarefa '{task_text}' tem só {len(ids_for_task)} "
                f"episódio(s) -- fica inteira em train, sem representação em val."
            )
            train_ids.extend(ids_for_task)
            continue
        tr, va = split_episodes(ids_for_task, val_frac, seed)
        train_ids.extend(tr)
        val_ids.extend(va)

    return train_ids, val_ids


def split_episodes_min_holdout(
    episode_ids: list[int],
    episode_task_labels: dict[int, str],
    n_val_per_task: int = 1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Reserva um número FIXO (não fração) de episódios por tarefa pra val.

    Diferente de split_episodes_stratified (fração): com poucos episódios
    por tarefa (Fase 2, ~5/tarefa), val_frac=0.1 arredonda de forma desigual
    entre tarefas (algumas ficam com 0 -- viram warning e vão inteiras pro
    train --, outras com 2). Fixar `n_val_per_task` (ex: 1) garante o MESMO
    número de episódios "nunca vistos" em toda tarefa que tiver dado
    suficiente -- métrica de val comparável entre tarefas, maximizando o
    quanto sobra pro treino.

    IMPORTANTE: o episódio reservado precisa ser genuinamente não visto no
    treino. Calcular a métrica de "validação" num episódio que o modelo já
    treinou mede memorização, não generalização -- não seria comparável ao
    val_recon_z0 de antes. Esta função garante essa separação; nunca compute
    a métrica de val "por fora" pegando episódios aleatórios do próprio
    train_dataset.

    Tarefas com só 1 episódio ficam inteiras em train (não há o que
    reservar sem zerar o treino dessa tarefa); aviso impresso nesse caso.
    Tarefas com exatamente `n_val_per_task` episódios reservam
    `n_val_per_task - 1` pra val, garantindo pelo menos 1 sobrando pro treino.
    """
    by_task: dict[str, list[int]] = defaultdict(list)
    for eid in episode_ids:
        by_task[episode_task_labels[eid]].append(eid)

    train_ids, val_ids = [], []
    rng = np.random.default_rng(seed=seed)
    for task_text, ids_for_task in by_task.items():
        ids = np.array(ids_for_task)
        rng.shuffle(ids)
        n_val = min(n_val_per_task, len(ids) - 1) if len(ids) > 1 else 0
        if n_val == 0:
            print(
                f"aviso: tarefa '{task_text}' tem só {len(ids)} episódio(s) "
                f"-- fica inteira em train, sem representação em val."
            )
        val_ids.extend(ids[:n_val].tolist())
        train_ids.extend(ids[n_val:].tolist())

    return train_ids, val_ids


def make_delta_timestamps(fps: float, obs_horizon: int = 1, pred_horizon: int = 50) -> dict:
    # O LiberoActBridge assume obs_horizon=1 (imagens SEM dimensão temporal,
    # comportamento observado do lerobot nesse caso). Com obs_horizon>1 o
    # stack das câmeras viraria (B, 2, T, C, H, W) e o ACT quebraria -- ou
    # pior, falharia de forma confusa. Falhar alto aqui, no ponto único de
    # consumo do parâmetro, até o bridge tratar a dimensão temporal.
    assert obs_horizon == 1, (
        f"obs_horizon={obs_horizon} não suportado: o LiberoActBridge assume "
        "obs_horizon=1 (sem dimensão temporal nas imagens). Ajuste o config "
        "ou estenda o bridge antes de mudar isso."
    )
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
