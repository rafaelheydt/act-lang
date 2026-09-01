"""Detecção de ambiente (Colab vs local) e seleção de device/checkpoint_dir.

Objetivo: o MESMO notebook roda sem edição tanto no Colab quanto numa máquina
local com GPU própria (ex: quando a sessão do Colab expira) -- só o que
precisa mudar entre os dois ambientes (montar Drive vs pasta local, device
único vs escolher entre várias GPUs) fica isolado aqui.
"""

import os
import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    """Seed global: torch (CPU+CUDA), numpy e random.

    Motivação (registro de método): o `seed` do config controlava SÓ o split
    de episódios -- init de pesos, dropout, amostragem do z e shuffle do
    DataLoader ficavam sem seed. Para comparar mecanismos de fusão, a
    diferença entre eles precisa ser maior que a variância entre runs; sem
    seed não dá nem pra estimar essa variância, nem pra garantir condições
    iniciais comparáveis. Para a comparação final, rode 2-3 seeds por
    mecanismo e reporte média ± desvio.

    NÃO liga o modo determinístico do cuDNN (torch.use_deterministic_
    algorithms): na T4 o custo de desempenho não compensa -- seed igual +
    mesmo hardware já dá reprodutibilidade prática suficiente aqui.
    """
    random.seed(seed)
    torch.manual_seed(seed)  # cobre CUDA também (torch >= 1.8)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def enable_fast_matmul() -> None:
    """Otimizações de matmul sem custo de precisão relevante para BC:

    - TF32 (Ampere+: RTX 30xx/A2000 em diante): matmuls fp32 ~2x mais
      rápidos com mantissa de 10 bits; em GPUs sem suporte (T4/Turing) as
      flags são ignoradas -- seguro chamar sempre.
    - cudnn.benchmark: shapes de entrada são FIXOS neste treino (mesma
      resolução, mesmo batch), então o autotuner do cuDNN escolhe o kernel
      mais rápido uma vez e reutiliza.
    Chamar uma vez, antes de criar o modelo."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def pick_device(preferred_index: int | None = None) -> torch.device:
    """Escolhe o device de treino.

    - `preferred_index` explícito sempre vence (ex: você quer forçar a
      RTX 3050 = índice 1 na sua máquina, mesmo com outra GPU disponível).
    - Sem preferência e com múltiplas GPUs: escolhe a com mais memória
      LIVRE no momento (via torch.cuda.mem_get_info) -- útil numa máquina
      compartilhada como a sua, com GPUs de tamanhos diferentes e outros
      processos (Xorg, gnome-shell) já ocupando um pouco de VRAM.
    - Sem CUDA disponível: cai pra CPU (não trava, só fica lento).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if preferred_index is not None:
        return torch.device(f"cuda:{preferred_index}")

    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        return torch.device("cuda:0")

    free_by_gpu = {}
    for i in range(n_gpus):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        free_by_gpu[i] = free_bytes
    best_index = max(free_by_gpu, key=free_by_gpu.get)
    return torch.device(f"cuda:{best_index}")


def describe_devices() -> str:
    """String legível com todas as GPUs visíveis e memória livre -- útil
    pra imprimir uma vez no início do notebook e confirmar a escolha."""
    if not torch.cuda.is_available():
        return "CUDA indisponível -- rodando em CPU."
    lines = []
    for i in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        lines.append(
            f"  cuda:{i} {name} -- {free_bytes / 1e9:.1f}GB livres de {total_bytes / 1e9:.1f}GB"
        )
    return "\n".join(lines)


def get_checkpoint_dir(experiment_name: str, local_base: str | Path | None = None) -> Path:
    """Colab: monta o Drive e devolve MyDrive/<experiment_name>.
    Local: usa `local_base` (ou $ACT_LANG_CHECKPOINT_DIR, ou ~/act-lang-checkpoints).
    """
    if is_colab():
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        checkpoint_dir = Path("/content/drive/MyDrive") / experiment_name
    else:
        base = local_base or os.environ.get("ACT_LANG_CHECKPOINT_DIR") or (Path.home() / "act-lang-checkpoints")
        checkpoint_dir = Path(base) / experiment_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir
