"""Loop de treino e avaliação do ACT.

CORREÇÕES aplicadas em relação ao notebook:
  - Validação determinística: z = mu (sample_posterior=False), sem ruído de
    reparametrização contaminando o critério de seleção de modelo.
  - Avaliação em UMA passada pelo val_loader: os dois forwards (z=mu e z=0)
    acontecem no mesmo batch -> o vídeo é decodificado uma vez, não duas.
  - Seleção de melhor checkpoint por `val_recon_z0` — a métrica que
    corresponde à inferência real (z=0, sem espiar as ações), não o
    val_loss (que mistura KL e mede outra coisa). Fiel ao ACT oficial:
    roda `num_epochs` fixas, sem early stopping (train_bc em
    imitate_episodes.py também não para antes da hora).

MUDANÇA (Fase 2, opcional): `fit(..., val_loader=None)` treina com TODOS os
dados (sem held-out). Justificativa: com poucos episódios por tarefa (~5,
10 tarefas), um split de validação fica pequeno demais pra ser um sinal
confiável -- e o que importa de verdade é taxa de sucesso no ambiente, não
L1 de reconstrução num punhado de episódios reservados. Sem val_loader não
há como fazer early stopping nem seleção de "melhor" checkpoint por métrica
offline: o treino roda por `num_epochs` fixo, salvando checkpoints
periódicos: a AVALIAÇÃO de qual é o melhor passa a ser feita depois, via
rollout real (ver eval/rollout_libero.py), não aqui.

DIAGNÓSTICO: `mu_abs_mean` (|mu| médio do posterior) é logado em cada época,
treino e val. Motivação: no treino do LIBERO tarefa única, as curvas
`val (z=mu)` e `val (z=0)` ficaram sobrepostas do início ao fim -- sinal de
posterior colapsado (mu ~ 0), mas isso era só uma inferência visual. Com
essa métrica, o colapso (ou não) fica numérico e visível época a época, sem
precisar comparar curvas de olho.
"""

import time
from pathlib import Path
from typing import Callable

import torch

from .checkpoints import save_checkpoint, save_top_k_checkpoint
from .loss import act_loss, kl_weight_schedule, masked_l1

Bridge = Callable  # (batch, device) -> (images, state, actions, is_pad, task_texts)


def train_one_epoch(
    model, loader, bridge: Bridge, optimizer, scaler, device,
    kl_weight: float, free_bits: float, grad_clip_norm: float = 10.0,
) -> dict:
    model.train()
    sums = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "mu_abs_mean": 0.0}
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
        sums["mu_abs_mean"] += mu.detach().abs().mean().item()
        n_batches += 1
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, bridge: Bridge, device, kl_weight: float, free_bits: float) -> dict:
    """Passada única: métricas com z=mu (determinístico) e recon com z=0."""
    model.eval()
    sums = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "recon_z0": 0.0, "mu_abs_mean": 0.0}
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
        sums["mu_abs_mean"] += mu.abs().mean().item()
        n_batches += 1
    model.train()
    return {k: v / n_batches for k, v in sums.items()}


def fit(
    model, train_loader, val_loader, bridge: Bridge, optimizer, device,
    checkpoint_dir: Path, num_epochs: int = 300, kl_weight: float = 10.0,
    kl_warmup_epochs: int = 0, free_bits: float = 0.0, grad_clip_norm: float = 10.0,
    checkpoint_every: int = 50, start_epoch: int = 0, history: dict | None = None,
) -> dict:
    """Treina o modelo.

    Roda `num_epochs` fixas, sempre — sem early stopping, igual ao ACT
    oficial (train_bc em imitate_episodes.py roda `range(num_epochs)`
    inteiro, sem `patience` nem parada antecipada).

    `kl_warmup_epochs > 0` ativa annealing linear do kl_weight (0 ->
    kl_weight, ao longo dessas épocas -- ver loss.kl_weight_schedule).
    Técnica de fora do ACT/paper de referência, usada para incentivar o
    decoder a de fato USAR z (diferente do free_bits, que só evita o
    encoder colapsar mu/logvar -- são dois mecanismos distintos, ver
    diagnóstico em diagnose_latent_usage.py). Padrão (0) desliga, kl_weight
    fica fixo como sempre foi.

    Com `val_loader` (comportamento original): seleção do melhor checkpoint
    por `val_recon_z0` (ainda salva os top-3, mas nunca interrompe o treino
    por falta de melhora).

    Com `val_loader=None` (Fase 2, dataset pequeno demais pra val confiável):
    treina com TODOS os dados por `num_epochs` fixo, salvando um checkpoint
    periódico a cada `checkpoint_every` épocas. A avaliação de qual desses
    checkpoints é o melhor fica pro rollout real, depois do treino.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    has_val = val_loader is not None

    if history is None:
        history = {}
    # setdefault (não dict fixo): robusto tanto a rodar sem val quanto a
    # retomar de um checkpoint salvo por uma versão anterior deste arquivo,
    # cujo history pode não ter todas as chaves de hoje (ex: mu_abs_mean).
    best_metric = min(history.get("val_recon_z0", []), default=float("inf"))
    top_k_checkpoints: list = []

    for epoch in range(start_epoch, num_epochs):
        current_kl_weight = kl_weight_schedule(epoch, kl_weight, kl_warmup_epochs)
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, bridge, optimizer, scaler, device,
            current_kl_weight, free_bits, grad_clip_norm,
        )
        for k, v in tr.items():
            history.setdefault(f"train_{k}", []).append(v)
        history.setdefault("kl_weight", []).append(current_kl_weight)

        log_line = (
            f"epoch {epoch + 1}/{num_epochs} | kl_w {current_kl_weight:.3f} | "
            f"train {tr['loss']:.4f} (recon {tr['recon']:.4f}, kld {tr['kld']:.5f}, "
            f"|mu| {tr['mu_abs_mean']:.4f})"
        )

        if has_val:
            va = evaluate(model, val_loader, bridge, device, current_kl_weight, free_bits)
            for k, v in va.items():
                history.setdefault(f"val_{k}", []).append(v)
            log_line += (
                f" | val {va['loss']:.4f} (recon {va['recon']:.4f}, kld {va['kld']:.5f}, "
                f"|mu| {va['mu_abs_mean']:.4f}) | z0 {va['recon_z0']:.4f}"
            )
        print(log_line + f" | {time.time() - t0:.1f}s")

        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", epoch, model, optimizer, history)

        if has_val and va["recon_z0"] < best_metric:
            # Critério de seleção: val_recon_z0 (fidelidade à inferência).
            # Só seleciona/salva o melhor — nunca interrompe o treino.
            best_metric = va["recon_z0"]
            top_k_checkpoints = save_top_k_checkpoint(
                checkpoint_dir, epoch, va["recon_z0"], model, optimizer,
                history, top_k_checkpoints, k=3, metric_name="z0",
            )
            print(f"  -> novo melhor val_recon_z0: {best_metric:.4f} | "
                  f"top-3: {[f'{v:.4f}' for v, _ in top_k_checkpoints]}")

        if (epoch + 1) % checkpoint_every == 0:
            fname = checkpoint_dir / f"periodic_epoch{epoch:03d}.pt"
            save_checkpoint(fname, epoch, model, optimizer, history)
            print(f"  -> checkpoint periódico (época {epoch + 1})")

    return history