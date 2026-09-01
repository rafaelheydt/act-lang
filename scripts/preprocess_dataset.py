"""Pré-processa o dataset LIBERO: decodifica os vídeos UMA vez e salva os
frames prontos em disco, no layout que data/preprocessed.py lê.

Motivação (diagnóstico de set/2026): época de horas no Colab/T4 com GPU
ociosa -- o custo era decodificar vídeo por amostra, a cada época. Depois
deste script, o treino lê JPEGs prontos e o gargalo volta a ser a GPU.

CORREÇÃO (01/09): antes deste patch, o loop instanciava um LeRobotDataset
NOVO por episódio (1693x) sem `root=` explícito -- cada instância parte de
um diretório "espelho" vazio (~/.cache/huggingface/lerobot/{repo}), então
`_load_metadata()`/`reader.try_load()` FALHAM sempre, mesmo com tudo já
baixado, disparando get_safe_version() + snapshot_download() -- CHAMADAS DE
API -- em TODA iteração. Com anônimo (~500 req/300s), estoura por volta do
episódio 20. Corrigido baixando o dataset UMA VEZ com snapshot_download e
passando `root=` explícito a toda instância: o load passa a ser 100% local,
sem nenhuma chamada de rede depois do download inicial (a doc do
LeRobotDataset confirma: "root... This can happen while you're offline").

Uso (uma vez por conjunto de tarefas; o das 40 serve também aos de 10,
porque o filtro por task acontece de novo na hora de treinar):

    python scripts/preprocess_dataset.py --config language_40_film \\
        --out data/preprocessed_libero40

    # depois, o treino rápido:
    python scripts/train.py --config language_40_film \\
        --preprocessed-dir data/preprocessed_libero40

Espaço em disco: ~20-40GB para as 40 tarefas (JPEG q92), MAIS ~10-20GB do
dataset bruto em --data-root (padrão: data/libero_raw/ -- baixado uma vez,
reaproveitado por qualquer --config depois). É retomável: cada episódio
ganha um marcador .done; re-rodar pula os já concluídos (sessão caiu no
meio -> rode de novo, ele continua).

Nota de fidelidade: JPEG é lossy (como o próprio vídeo de origem já é);
q92 é visualmente indistinguível e é prática padrão. Para bit-exatidão,
--format png (~3-4x mais disco, leitura um pouco mais lenta).
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from act_lang.data.libero import (
    REPO_ID, filter_episodes_by_tasks, get_episode_task_labels,
)
from act_lang.data.preprocessed import (
    ARRAYS_NAME, FRAMES_DIR, META_NAME, PREPROCESS_VERSION,
)
from train import CONFIG_REGISTRY, load_config  # scripts/ está no sys.path[0]


def _to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] -> (H, W, C) uint8."""
    return (img.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).numpy()


def _stats_to_json(stats: dict) -> dict:
    """Copia min/max de state/action dos stats do lerobot para o meta.json.

    Guardar os stats GLOBAIS aqui garante normalização idêntica entre o
    caminho lerobot (Colab) e o caminho pré-processado (local) -- checkpoints
    continuam intercambiáveis.
    """
    out = {}
    for key in ("observation.state", "action"):
        out[key] = {
            "min": np.asarray(stats[key]["min"]).astype(float).ravel().tolist(),
            "max": np.asarray(stats[key]["max"]).astype(float).ravel().tolist(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIG_REGISTRY))
    parser.add_argument("--out", required=True, type=Path,
                        help="Pasta de saída (criada se não existir).")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=92,
                        help="Qualidade JPEG (ignorado com --format png).")
    parser.add_argument("--data-root", type=Path, default=Path("data/libero_raw"),
                        help="Onde baixar/ler o dataset lerobot/libero completo "
                             "(uma vez; reaproveitado em runs futuros e por "
                             "qualquer --config). Separado de --out, que é onde "
                             "vão os FRAMES pré-processados.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out: Path = args.out
    (out / FRAMES_DIR).mkdir(parents=True, exist_ok=True)

    # Download ÚNICO do dataset inteiro (meta + data + vídeos). Só faz
    # trabalho de rede pros arquivos que ainda não estão em --data-root;
    # re-rodar (mesmo config, ou outro --config sobre o mesmo --data-root)
    # não baixa nada de novo. É esta chamada, sozinha, que substitui as
    # 1693 chamadas de API que o loop antigo fazia.
    print(f"baixando/verificando lerobot/{REPO_ID.split('/')[-1]} em {args.data_root} "
          "(pula o que já estiver completo)...")
    args.data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.data_root)

    meta_lr = LeRobotDatasetMetadata(REPO_ID, root=args.data_root)
    full_dataset = LeRobotDataset(REPO_ID, root=args.data_root)
    episode_ids = filter_episodes_by_tasks(meta_lr, full_dataset, cfg["task_texts"])
    labels = get_episode_task_labels(meta_lr, full_dataset, episode_ids)
    print(f"{len(episode_ids)} episódios a pré-processar -> {out}")

    # retomada: arrays já salvos de uma rodada interrompida
    arrays_path = out / ARRAYS_NAME
    arrays: dict = {}
    if arrays_path.exists():
        arrays = torch.load(arrays_path, map_location="cpu", weights_only=False)
        print(f"retomando: {len(arrays)} episódios já em {ARRAYS_NAME}")

    episodes_meta: dict = {}
    for n, ep in enumerate(sorted(episode_ids), 1):
        ep_dir = out / FRAMES_DIR / f"ep{ep:05d}"
        done_flag = ep_dir / ".done"
        if done_flag.exists() and ep in arrays:
            episodes_meta[str(ep)] = {
                "task": labels[ep], "length": arrays[ep]["state"].shape[0],
            }
            continue

        ep_dir.mkdir(parents=True, exist_ok=True)
        # root= explícito: lê 100% do disco local já baixado, ZERO chamadas
        # de rede (sem isto, cada instância parte de um diretório "espelho"
        # vazio e recorre à API do Hub em toda iteração -- ver docstring).
        ds = LeRobotDataset(REPO_ID, root=args.data_root, episodes=[ep])  # sem delta_ts: 1 item = 1 frame
        states, actions = [], []
        for t in range(len(ds)):
            item = ds[t]
            for cam, key in enumerate(
                ("observation.images.image", "observation.images.image2")
            ):
                img = Image.fromarray(_to_uint8_hwc(item[key]))
                fp = ep_dir / f"c{cam}_{t:05d}.{args.format}"
                if args.format == "jpg":
                    img.save(fp, quality=args.quality)
                else:
                    img.save(fp)
            states.append(item["observation.state"].float().reshape(-1))
            actions.append(item["action"].float().reshape(-1))

        arrays[ep] = {
            "state": torch.stack(states), "action": torch.stack(actions),
        }
        episodes_meta[str(ep)] = {"task": labels[ep], "length": len(states)}
        # arrays + marcador salvos POR episódio: interrompeu, não perde nada
        torch.save(arrays, arrays_path)
        done_flag.touch()
        print(f"[{n}/{len(episode_ids)}] ep {ep}: {len(states)} frames "
              f"({labels[ep][:50]}...)")

    meta = {
        "version": PREPROCESS_VERSION,
        "source_repo_id": REPO_ID,
        "fps": float(meta_lr.fps),
        "image_format": args.format,
        "stats": _stats_to_json(meta_lr.stats),
        "episodes": episodes_meta,
    }
    (out / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    n_frames = sum(v["length"] for v in episodes_meta.values())
    print(f"concluído: {len(episodes_meta)} episódios, {n_frames} frames, "
          f"meta em {out / META_NAME}")


if __name__ == "__main__":
    main()
