"""Loop de treino e avaliação do ACT.

CORREÇÕES aplicadas em relação ao notebook:
  - Validação determinística: z = mu (sample_posterior=False), sem ruído de
    reparametrização contaminando o critério de seleção de modelo.
  - Avaliação em UMA passada pelo val_loader: os dois forwards (z=mu e z=0)
    acontecem no mesmo batch -> o vídeo é decodificado uma vez, não duas.
  - Early stopping e seleção de melhor checkpoint por `val_recon_z0` — a
    métrica que corresponde à inferência real (z=0, sem espiar as ações),
    não o val_loss (que mistura KL e mede outra coisa).
"""

import time
from pathlib import Path
from typing import Callable

import torch

from .checkpoints import save_checkpoint, save_top_k_checkpoint
from .loss import act_loss, masked_l1

Bridge = Callable  # (batch, device) -> (images, state, actions, is_pad, task_texts)


def train_one_epoch(
    model, loader, bridge: Bridge, optimizer, scaler, device,
    kl_weight: float, free_bits: float, grad_clip_norm: float = 10.0,
) -> dict:
    model.train()
    sums = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
    n_batches = 0
    for batch in loader:
        images, state, actions, is_pad, task_texts = bridge(batch, device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred_actions, mu, logvar = model(
                images, state, actions=actions, is_pad=is_pad, task_texts=task_texts
            )
            loss, recon, kld = act_loss(
                pred_actions, actions, mu, logvar, is_pad, kl_weight, free_bits
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale ANTES do clip — padrão AMP correto
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        sums["loss"] += loss.item()
        sums["recon"] += recon.item()
        sums["kld"] += kld.item()
        n_batches += 1
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, bridge: Bridge, device, kl_weight: float, free_bits: float) -> dict:
    """Passada única: métricas com z=mu (determinístico) e recon com z=0."""
    model.eval()
    sums = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "recon_z0": 0.0}
    n_batches = 0
    for batch in loader:
        images, state, actions, is_pad, task_texts = bridge(batch, device)

        # posterior determinístico (z = mu)
        pred_actions, mu, logvar = model(
            images, state, actions=actions, is_pad=is_pad,
            task_texts=task_texts, sample_posterior=False,
        )
        loss, recon, kld = act_loss(
            pred_actions, actions, mu, logvar, is_pad, kl_weight, free_bits
        )

        # inferência real (z = 0) — a métrica que prevê o rollout
        pred_z0, _, _ = model(images, state, actions=None, task_texts=task_texts)
        recon_z0 = masked_l1(pred_z0, actions, is_pad)

        sums["loss"] += loss.item()
        sums["recon"] += recon.item()
        sums["kld"] += kld.item()
        sums["recon_z0"] += recon_z0.item()
        n_batches += 1
    model.train()
    return {k: v / n_batches for k, v in sums.items()}


def fit(
    model, train_loader, val_loader, bridge: Bridge, optimizer, device,
    checkpoint_dir: Path, num_epochs: int = 300, kl_weight: float = 10.0,
    free_bits: float = 0.05, grad_clip_norm: float = 10.0, patience: int = 40,
    checkpoint_every: int = 50, start_epoch: int = 0, history: dict | None = None,
) -> dict:
    """Treina com early stopping e seleção de melhor checkpoint por val_recon_z0."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    if history is None:
        history = {k: [] for k in (
            "train_loss", "train_recon", "train_kld",
            "val_loss", "val_recon", "val_kld", "val_recon_z0",
        )}
    best_metric = min(history["val_recon_z0"], default=float("inf"))
    epochs_without_improvement = 0
    top_k_checkpoints: list = []

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, bridge, optimizer, scaler, device,
            kl_weight, free_bits, grad_clip_norm,
        )
        va = evaluate(model, val_loader, bridge, device, kl_weight, free_bits)

        for k, v in tr.items():
            history[f"train_{k}"].append(v)
        for k, v in va.items():
            history[f"val_{k}"].append(v)

        print(
            f"epoch {epoch + 1}/{num_epochs} | "
            f"train {tr['loss']:.4f} (recon {tr['recon']:.4f}, kld {tr['kld']:.5f}) | "
            f"val {va['loss']:.4f} (recon {va['recon']:.4f}, kld {va['kld']:.5f}) | "
            f"z0 {va['recon_z0']:.4f} | {time.time() - t0:.1f}s"
        )

        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", epoch, model, optimizer, history)

        # Critério de seleção: val_recon_z0 (fidelidade à inferência).
        if va["recon_z0"] < best_metric:
            best_metric = va["recon_z0"]
            epochs_without_improvement = 0
            top_k_checkpoints = save_top_k_checkpoint(
                checkpoint_dir, epoch, va["recon_z0"], model, optimizer,
                history, top_k_checkpoints, k=3, metric_name="z0",
            )
            print(f"  -> novo melhor val_recon_z0: {best_metric:.4f} | "
                  f"top-3: {[f'{v:.4f}' for v, _ in top_k_checkpoints]}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Sem melhora por {patience} épocas — early stopping.")
                break

        if (epoch + 1) % checkpoint_every == 0:
            fname = checkpoint_dir / f"periodic_epoch{epoch:03d}.pt"
            save_checkpoint(fname, epoch, model, optimizer, history)
            print(f"  -> checkpoint periódico (época {epoch + 1})")

    return history
