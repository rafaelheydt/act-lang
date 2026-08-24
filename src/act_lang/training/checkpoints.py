"""Salvamento/retomada de checkpoints e ranking top-k.

CORREÇÃO aplicada: `load_checkpoint` retorna `epoch + 1` como start_epoch —
o notebook original re-executava a época já salva ao retomar (off-by-one).
"""

from pathlib import Path

import torch


def save_checkpoint(path: Path, epoch: int, model, optimizer, history: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, device="cpu") -> tuple[int, dict]:
    """Retorna (start_epoch, history) — start_epoch é a PRÓXIMA época a rodar."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"] + 1, ckpt["history"]


def save_top_k_checkpoint(
    checkpoint_dir: Path, epoch: int, metric_value: float, model, optimizer,
    history: dict, top_k_list: list, k: int = 3, metric_name: str = "z0",
) -> list:
    fname = checkpoint_dir / f"best_epoch{epoch:03d}_{metric_name}{metric_value:.4f}.pt"
    save_checkpoint(fname, epoch, model, optimizer, history)
    top_k_list.append((metric_value, fname))
    top_k_list.sort(key=lambda x: x[0])
    if len(top_k_list) > k:
        _, removed_fname = top_k_list.pop()
        if removed_fname.exists():
            removed_fname.unlink()
    return top_k_list
