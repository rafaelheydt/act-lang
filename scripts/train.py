"""Script de treino standalone (equivalente a notebooks/01_treino_libero.ipynb,
sem a parte específica de Colab). Uso local/servidor:

    python scripts/train.py --config single_task
    python scripts/train.py --config language_film --resume

Pré-requisito (uma vez, no terminal, dentro do seu ambiente):
    pip install -e ".[language]" "lerobot[libero]"
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

# scripts/ fica FORA de src/ de propósito (mesmo motivo do configs/ — ver
# comentário em configs/libero_single_task.py), mas isso significa que
# `sys.path[0]` (a pasta deste arquivo) NÃO inclui a raiz do repositório
# quando você roda `python scripts/train.py` -- sem esta linha,
# `import configs.libero_single_task` falha com ModuleNotFoundError,
# mesmo rodando do lugar certo. Usa __file__, não o cwd: funciona não
# importa de onde você chame o script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from act_lang.data.libero import (
    REPO_ID, LiberoActBridge, filter_episodes_by_tasks, get_episode_task_labels,
    make_delta_timestamps, split_episodes_min_holdout, split_episodes_stratified,
)
from act_lang.data.normalize import MinMaxNormalizer
from act_lang.models.act import ACT
from act_lang.models.backbone import freeze_batchnorm
from act_lang.models.fusion import build_fusion
from act_lang.training.checkpoints import load_checkpoint
from act_lang.training.loop import fit
from act_lang.training.optim import build_optimizer
from act_lang.data.preprocessed import PreprocessedLiberoDataset, build_normalizers_from_meta
from act_lang.data.hdf5_libero import (
    HDF5LiberoDataset, load_meta as load_meta_hdf5,
    build_normalizers_from_meta as build_normalizers_from_meta_hdf5,
)
from act_lang.utils.runtime import (
    describe_devices, enable_fast_matmul, get_checkpoint_dir, is_colab,
    pick_device, set_seed,
)

# nome CLI -> (módulo em configs/, atributo do dict CONFIG dentro dele)
CONFIG_REGISTRY = {
    "single_task": ("configs.libero_single_task", "CONFIG"),
    "multitask": ("configs.libero_object_multitask", "CONFIG"),
    "language_token": ("configs.libero_object_language", "CONFIG_TOKEN"),
    "language_film": ("configs.libero_object_language", "CONFIG_FILM"),
    "language_cross_attn": ("configs.libero_object_language", "CONFIG_CROSS_ATTN"),
    "language_40_token": ("configs.libero_40tasks_language", "CONFIG_TOKEN"),
    "language_40_film": ("configs.libero_40tasks_language", "CONFIG_FILM"),
    "language_40_cross_attn": ("configs.libero_40tasks_language", "CONFIG_CROSS_ATTN"),
}


def load_config(name: str) -> dict:
    module_path, attr = CONFIG_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _loader_kwargs(cfg: dict, device: torch.device) -> dict:
    """Knobs de DataLoader por ambiente. No Colab, workers com decode de
    vídeo travam (fork+ffmpeg) -> 0. Fora dele, workers de verdade: com os
    frames pré-processados, 8 workers numa CPU de 16 núcleos saturam a GPU;
    override explícito via cfg["num_workers"] se precisar."""
    if "num_workers" in cfg:
        nw = cfg["num_workers"]
    elif is_colab():
        nw = 0
    else:
        nw = min(8, os.cpu_count() or 1)
    kwargs = dict(num_workers=nw, pin_memory=(device.type == "cuda"))
    if nw > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return kwargs


def build_data_preprocessed(cfg: dict, device: torch.device, root):
    """Caminho RÁPIDO: frames pré-processados em disco (ver
    scripts/preprocess_dataset.py e data/preprocessed.py). Mesmos splits,
    mesma normalização (stats globais copiados do lerobot), mesmo bridge --
    só a origem dos bytes muda."""
    probe = PreprocessedLiberoDataset(root, cfg["pred_horizon"], cfg["task_texts"])
    labels = probe.episode_task_labels
    episode_ids = sorted(labels)

    val_strategy = cfg.get("val_strategy", "fraction")
    if val_strategy == "min_holdout":
        train_ids, val_ids = split_episodes_min_holdout(
            episode_ids, labels, cfg.get("n_val_per_task", 1), cfg["seed"]
        )
    else:
        train_ids, val_ids = split_episodes_stratified(
            episode_ids, labels, cfg["val_frac"], cfg["seed"]
        )
    print(f"[pré-processado: {root}] episódios: {len(episode_ids)} -> "
          f"train {len(train_ids)} | val {len(val_ids)} (val_strategy={val_strategy!r})")

    train_dataset = PreprocessedLiberoDataset(
        root, cfg["pred_horizon"], cfg["task_texts"], episodes=train_ids
    )
    val_dataset = PreprocessedLiberoDataset(
        root, cfg["pred_horizon"], cfg["task_texts"], episodes=val_ids
    )
    kwargs = _loader_kwargs(cfg, device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, **kwargs,
    )
    state_norm, action_norm = build_normalizers_from_meta(train_dataset.meta, device)
    bridge = LiberoActBridge(state_norm, action_norm)
    return train_loader, val_loader, bridge


def build_data_hdf5(cfg: dict, device: torch.device, root):
    """Caminho HDF5 nativo: arrays uint8 crus do LIBERO oficial, sem
    decodificação de imagem nenhuma (nem vídeo, nem JPEG) -- ver
    docs/hdf5_migration.md pro raciocínio completo e scripts/download_libero_hdf5.py
    pra baixar `root`. Mesmos splits, mesmo bridge; resolução nativa 128x128
    (decisão tomada com o usuário -- ver docs/hdf5_migration.md)."""
    probe = HDF5LiberoDataset(root, cfg["pred_horizon"], cfg["task_texts"])
    labels = probe.episode_task_labels
    episode_ids = sorted(labels)

    val_strategy = cfg.get("val_strategy", "fraction")
    if val_strategy == "min_holdout":
        train_ids, val_ids = split_episodes_min_holdout(
            episode_ids, labels, cfg.get("n_val_per_task", 1), cfg["seed"]
        )
    else:
        train_ids, val_ids = split_episodes_stratified(
            episode_ids, labels, cfg["val_frac"], cfg["seed"]
        )
    print(f"[hdf5: {root}] episódios: {len(episode_ids)} -> "
          f"train {len(train_ids)} | val {len(val_ids)} (val_strategy={val_strategy!r})")

    train_dataset = HDF5LiberoDataset(
        root, cfg["pred_horizon"], cfg["task_texts"], episodes=train_ids
    )
    val_dataset = HDF5LiberoDataset(
        root, cfg["pred_horizon"], cfg["task_texts"], episodes=val_ids
    )
    kwargs = _loader_kwargs(cfg, device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, **kwargs,
    )
    meta = load_meta_hdf5(root)
    state_norm, action_norm = build_normalizers_from_meta_hdf5(meta, device)
    bridge = LiberoActBridge(state_norm, action_norm)
    return train_loader, val_loader, bridge


def build_data(cfg: dict, device: torch.device):
    meta = LeRobotDatasetMetadata(REPO_ID)
    full_dataset = LeRobotDataset(REPO_ID)

    episode_ids = filter_episodes_by_tasks(meta, full_dataset, cfg["task_texts"])
    episode_task_labels = get_episode_task_labels(meta, full_dataset, episode_ids)

    val_strategy = cfg.get("val_strategy", "fraction")
    if val_strategy == "min_holdout":
        train_ids, val_ids = split_episodes_min_holdout(
            episode_ids, episode_task_labels, cfg.get("n_val_per_task", 1), cfg["seed"]
        )
    else:
        train_ids, val_ids = split_episodes_stratified(
            episode_ids, episode_task_labels, cfg["val_frac"], cfg["seed"]
        )
    print(f"episódios: {len(episode_ids)} -> train {len(train_ids)} | val {len(val_ids)} "
          f"(val_strategy={val_strategy!r})")

    delta_ts = make_delta_timestamps(meta.fps, cfg["obs_horizon"], cfg["pred_horizon"])
    train_dataset = LeRobotDataset(REPO_ID, episodes=train_ids, delta_timestamps=delta_ts)
    val_dataset = LeRobotDataset(REPO_ID, episodes=val_ids, delta_timestamps=delta_ts)

    kwargs = _loader_kwargs(cfg, device)  # 0 workers no Colab; 8 fora dele
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, **kwargs,
    )

    state_norm = MinMaxNormalizer.from_lerobot_stats(meta.stats, "observation.state").to(device)
    action_norm = MinMaxNormalizer.from_lerobot_stats(meta.stats, "action").to(device)
    bridge = LiberoActBridge(state_norm, action_norm)
    return train_loader, val_loader, bridge


def build_model_and_optimizer(cfg: dict, device: torch.device):
    fusion = build_fusion(cfg.get("fusion_type"), d_model=cfg["d_model"])
    if fusion is not None:
        print(f"fusão de linguagem: {cfg['fusion_type']}")

    model = ACT(
        action_dim=cfg["action_dim"], state_dim=cfg["state_dim"],
        d_model=cfg["d_model"], latent_dim=cfg["latent_dim"],
        chunk_size=cfg["chunk_size"], n_cameras=cfg["n_cameras"],
        n_encoder_layers=cfg["n_encoder_layers"], n_decoder_layers=cfg["n_decoder_layers"],
        n_heads=cfg["n_heads"], dropout=cfg["dropout"], pretrained_backbone=True,
        decoder_style=cfg["decoder_style"], fusion=fusion,
    )
    if cfg["freeze_bn"]:
        freeze_batchnorm(model.vision_backbone)
    model = model.to(device)
    print(f"parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = build_optimizer(model, cfg["lr"], cfg["lr_backbone"], cfg["weight_decay"])
    return model, optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIG_REGISTRY))
    parser.add_argument("--resume", action="store_true",
                         help="Retoma de checkpoint_dir/last_checkpoint.pt, se existir.")
    parser.add_argument("--preprocessed-dir", type=Path, default=None,
                         help="Pasta gerada por scripts/preprocess_dataset.py; "
                              "se passada, treina lendo frames prontos do disco "
                              "(ordens de magnitude mais rápido que decodificar vídeo).")
    parser.add_argument("--hdf5-dir", type=Path, default=None,
                         help="Pasta gerada por scripts/download_libero_hdf5.py; "
                              "se passada, treina lendo HDF5 nativo (128x128, sem "
                              "decodificação de imagem nenhuma -- ver "
                              "docs/hdf5_migration.md). Mutuamente exclusivo com "
                              "--preprocessed-dir.")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Sobrescreve o diretório de checkpoints (padrão: get_checkpoint_dir).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"config: {args.config} (experiment_name={cfg['experiment_name']!r})")

    print(describe_devices())
    device = pick_device(preferred_index=cfg.get("device_index"))
    print(f"usando: {device}")

    set_seed(cfg["seed"])  # pesos, dropout, z, shuffle -- não só o split
    enable_fast_matmul()   # TF32 (Ampere+) + cudnn.benchmark; no-op na T4

    assert not (args.preprocessed_dir is not None and args.hdf5_dir is not None), (
        "--preprocessed-dir e --hdf5-dir são mutuamente exclusivos -- escolha uma origem de dados."
    )
    if args.hdf5_dir is not None:
        train_loader, val_loader, bridge = build_data_hdf5(cfg, device, args.hdf5_dir)
    elif args.preprocessed_dir is not None:
        train_loader, val_loader, bridge = build_data_preprocessed(
            cfg, device, args.preprocessed_dir
        )
    else:
        train_loader, val_loader, bridge = build_data(cfg, device)
    model, optimizer = build_model_and_optimizer(cfg, device)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else get_checkpoint_dir(cfg["experiment_name"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints em: {checkpoint_dir}")

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch, history = 0, None
    last_ckpt = checkpoint_dir / "last_checkpoint.pt"
    if args.resume:
        if last_ckpt.exists():
            start_epoch, history = load_checkpoint(
                last_ckpt, model, optimizer, device, scaler=scaler
            )
            print(f"retomando de {last_ckpt} -> start_epoch={start_epoch}")
        else:
            print(f"--resume passado, mas {last_ckpt} não existe -- começando do zero.")

    fit(
        model, train_loader, val_loader, bridge, optimizer, device,
        checkpoint_dir=checkpoint_dir, num_epochs=cfg["num_epochs"],
        kl_weight=cfg["kl_weight"],
        kl_warmup_epochs=cfg.get("kl_warmup_epochs", 0),  # antes: silenciosamente ignorado
        free_bits=cfg["free_bits"],
        grad_clip_norm=cfg["grad_clip_norm"], checkpoint_every=cfg["checkpoint_every"],
        start_epoch=start_epoch, history=history, scaler=scaler,
        accum_steps=cfg.get("accum_steps", 1),  # batch efetivo = batch_size x accum
    )


if __name__ == "__main__":
    main()