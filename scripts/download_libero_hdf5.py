"""Baixa os arquivos HDF5 oficiais do LIBERO (yifengzhu-hf/LIBERO-datasets,
mirror do dataset oficial Lifelong-Robot-Learning/LIBERO) pras 40 tarefas de
configs/task_mapping_libero40.json, e computa os stats (min/max de
action/observation.state) pro meta.json que HDF5LiberoDataset e
build_normalizers_from_meta (data/hdf5_libero.py) leem.

Ver docs/hdf5_migration.md pro raciocínio completo da migração: benchmarks
de I/O que motivaram trocar JPEG por HDF5 nativo, o mapeamento das 40
tarefas pros arquivos oficiais (task_mapping_libero40.json) e a verificação
de convenção de ação/imagem (por que NÃO precisa converter ação, e por que
as imagens do HDF5 precisam de flip vertical).

Espaço em disco: arquivos ~700-800MB cada (128x128, sem compressão), 40
tarefas -> ~28-32GB. Rode no Colab/Drive, não numa máquina de dev comum.
É retomável: hf_hub_download pula arquivos já baixados e íntegros (verifica
por hash).

Uso:
    python scripts/download_libero_hdf5.py --out data/libero_hdf5

    # baixar sem computar stats ainda (ex: quer rodar em duas sessões
    # porque a sessão do Colab caiu no meio do download):
    python scripts/download_libero_hdf5.py --out data/libero_hdf5 --skip-stats
    # depois, só stats (não rebaixa nada já presente):
    python scripts/download_libero_hdf5.py --out data/libero_hdf5

    # piloto rápido: só as N primeiras tarefas (ordem alfabética,
    # determinística) -- ~700-800MB x N em vez de ~28-32GB. root/task_mapping.json
    # fica com só essas N entradas, então HDF5LiberoDataset/train.py já
    # treinam só nelas automaticamente, mesmo com cfg["task_texts"] = as 40.
    python scripts/download_libero_hdf5.py --out data/libero_hdf5_piloto --limit 3
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from act_lang.data.hdf5_libero import (
    HDF5_VERSION, META_NAME, TASK_MAPPING_NAME, compute_stats, load_task_mapping,
)

TASK_MAPPING_SRC = _REPO_ROOT / "configs" / "task_mapping_libero40.json"
HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path,
                         help="Pasta de saída (criada se não existir).")
    parser.add_argument("--skip-stats", action="store_true",
                         help="Só baixa os arquivos, não escreve meta.json "
                              "(o scan de stats lê os 40 arquivos inteiros -- "
                              "útil separar do download em sessões curtas).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Baixa só as N primeiras tarefas (ordem alfabética "
                              "do texto da tarefa, determinística) em vez das 40 -- "
                              "pra um piloto rápido antes de comprometer ~28-32GB. "
                              "root/task_mapping.json fica só com essas N entradas.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    full_mapping = json.loads(TASK_MAPPING_SRC.read_text(encoding="utf-8"))["tasks"]
    if args.limit is not None:
        chosen = dict(sorted(full_mapping.items())[: args.limit])
        assert chosen, f"--limit {args.limit} não deixou nenhuma tarefa"
    else:
        chosen = full_mapping
    (args.out / TASK_MAPPING_NAME).write_text(
        json.dumps({"tasks": chosen}, ensure_ascii=False, indent=1)
    )
    mapping = load_task_mapping(args.out)

    print(f"{len(mapping)} tarefas -> baixando de {HF_REPO_ID} em {args.out} "
          "(pula o que já estiver completo)...")
    for n, (task, (suite, fname)) in enumerate(sorted(mapping.items()), 1):
        hf_hub_download(
            repo_id=HF_REPO_ID, repo_type="dataset", filename=f"{suite}/{fname}",
            local_dir=args.out,
        )
        print(f"[{n}/{len(mapping)}] {suite}/{fname} ({task[:60]}...)")

    if args.skip_stats:
        print("stats puladas (--skip-stats); rode de novo sem a flag pra gerar o meta.json.")
        return

    print(f"computando stats (min/max de action/observation.state, escaneando os {len(mapping)} arquivos)...")
    stats = compute_stats(args.out, mapping)
    meta = {"version": HDF5_VERSION, "resolution": 128, "stats": stats}
    (args.out / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"concluído: meta em {args.out / META_NAME}")


if __name__ == "__main__":
    main()
