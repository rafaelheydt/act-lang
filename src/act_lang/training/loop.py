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

CORREÇÕES (set/2026):
  - Top-k DE VERDADE: um checkpoint entra no ranking sempre que sua métrica
    é melhor que a do PIOR membro atual (ou o ranking está incompleto) — não
    apenas em novo recorde global. Antes, a época com o 2º melhor valor da
    corrida inteira era descartada se não tivesse sido recorde no momento em
    que ocorreu (ex: sequência 5.0 -> 4.0 -> 4.1 guardava 5.0 e 4.0, nunca
    4.1). Como o fluxo é escolher checkpoints pra rollout DEPOIS, o conjunto
    de candidatos nascia errado.
  - Resume completo: estado do GradScaler viaja no checkpoint, e o ranking
    top-k é reconstruído dos arquivos best_*.pt no disco (scan_top_k) — ver
    checkpoints.py.
  - evaluate() com agregação PONDERADA (somas acumuladas / divisão única no
    final): média-de-médias por batch enviesava com último batch menor e
    padding variável. recon/recon_z0 usam soma de erro / soma de elementos
    válidos; loss/kld/mu_abs_mean são ponderados pelo nº de amostras.
  - evaluate() roda sob autocast (mesma precisão do treino, ~metade do custo
    na T4 — são DOIS forwards por batch) e restaura o modo (train/eval) em
    que o modelo estava, em vez de forçar train() incondicional.

MUDANÇA (Fase 2, opcional): `fit(..., val_loader=None)` treina com TODOS os
dados (sem held-out). Justificativa: com poucos episódios por tarefa (~5,
10 tarefas), um split de validação fica pequeno demais pra ser um sinal
confiável -- e o que importa de verdade é taxa de sucesso no ambiente, não
L1 de reconstrução num punhado de episódios reservados. Sem val_loader não
há como fazer early stopping nem seleção de "melhor" checkpoint por métrica
offline: o treino roda por `num_epochs` fixo, salvando checkpoints
periódicos: a AVALIAÇÃO de qual é o melhor passa a ser feita depois, via
rollout real (ver eval/rollout_libero.py), não aqui.

USO DO VAL (registro de método): val_recon_z0 é um FILTRO de candidatos, não
um veredito — a correlação entre métrica offline e taxa de sucesso em
rollout é fraca (robomimic, Mandlekar et al. 2021). Decida entre os top-k e
o last por rollout, com init_state_ids fixos e compartilhados entre
experimentos (ver eval/rollout_libero.py).

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

from .checkpoints import save_checkpoint, save_top_k_checkpoint, scan_top_k
from .loss import act_loss, kl_weight_schedule, masked_l1_sums

Bridge = Callable  # (batch, device) -> (images, state, actions, is_pad, task_texts)

TOP_K = 3


def train_one_epoch(
    model, loader, bridge: Bridge, optimizer, scaler, device,
    kl_weight: float, free_bits: float, grad_clip_norm: float = 10.0,
    accum_steps: int = 1,
) -> dict:
    """`accum_steps > 1` ativa gradient accumulation: o step do optimizer
    acontece a cada accum_steps microbatches, com a loss dividida por
    accum_steps -- batch EFETIVO = batch_size x accum_steps. Motivação:
    manter o batch efetivo 32 da T4/Colab ao treinar em GPUs com menos
    VRAM (RTX 3050 8GB: batch 16 x accum 2; A2000 6GB: batch 8 x accum 4),
    senão o regime de treino vira um confundidor na comparação entre
    mecanismos de fusão. Sobra de microbatches no fim da época (época não
    múltipla de accum_steps) é aplicada num step final com o que houver.
    RESSALVA: a equivalência exata com o batch grande exige BatchNorm
    congelada (batch stats acoplam as amostras do microbatch) -- vale aqui
    porque todos os configs usam freeze_bn=True; se um dia freeze_bn sair,
    accumulation deixa de ser matematicamente idêntica (aproximação usual).
    As métricas logadas são as losses CHEIAS por microbatch (sem a divisão),
    comparáveis às de accum_steps=1."""
    model.train()
    sums = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "mu_abs_mean": 0.0}
    n_batches = 0

    def _apply_step():
        scaler.unscale_(optimizer)  # unscale ANTES do clip — padrão AMP correto
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    optimizer.zero_grad()
    pending = False
    for i, batch in enumerate(loader):
        images, state, actions, is_pad, task_texts = bridge(batch, device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            pred_actions, mu, logvar = model(
                images, state, actions=actions, is_pad=is_pad, task_texts=task_texts
            )
            loss, recon, kld = act_loss(
                pred_actions, actions, mu, logvar, is_pad, kl_weight, free_bits
            )

        scaler.scale(loss / accum_steps).backward()
        pending = True
        if (i + 1) % accum_steps == 0:
            _apply_step()
            pending = False

        sums["loss"] += loss.item()
        sums["recon"] += recon.item()
        sums["kld"] += kld.item()
        sums["mu_abs_mean"] += mu.detach().abs().mean().item()
        n_batches += 1
    if pending:  # microbatches restantes de época não múltipla de accum_steps
        _apply_step()
    return {k: v / n_batches for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, bridge: Bridge, device, kl_weight: float, free_bits: float) -> dict:
    """Passada única: métricas com z=mu (determinístico) e recon com z=0.

    Agregação ponderada (ver docstring do módulo): recon e recon_z0 por soma
    de erro / soma de elementos válidos; loss, kld e mu_abs_mean ponderados
    por nº de amostras. Restaura o modo (train/eval) de entrada ao sair.
    """
    was_training = model.training
    model.eval()
    err_sum = err_z0_sum = 0.0
    n_valid = n_valid_z0 = 0
    loss_sum = kld_sum = mu_sum = 0.0
    n_samples = 0
    for batch in loader:
        images, state, actions, is_pad, task_texts = bridge(batch, device)
        bsz = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # posterior determinístico (z = mu)
            pred_actions, mu, logvar = model(
                images, state, actions=actions, is_pad=is_pad,
                task_texts=task_texts, sample_posterior=False,
            )
            loss, _, kld = act_loss(
                pred_actions, actions, mu, logvar, is_pad, kl_weight, free_bits
            )
            e, n = masked_l1_sums(pred_actions, actions, is_pad)

            # inferência real (z = 0) — a métrica que prevê o rollout
            pred_z0, _, _ = model(images, state, actions=None, task_texts=task_texts)
            e_z0, n_z0 = masked_l1_sums(pred_z0, actions, is_pad)

        err_sum += e.item()
        n_valid += n.item()
        err_z0_sum += e_z0.item()
        n_valid_z0 += n_z0.item()
        loss_sum += loss.item() * bsz
        kld_sum += kld.item() * bsz
        mu_sum += mu.abs().mean().item() * bsz
        n_samples += bsz

    if was_training:
        model.train()
    return {
        "loss": loss_sum / max(n_samples, 1),
        "recon": err_sum / max(n_valid, 1),
        "kld": kld_sum / max(n_samples, 1),
        "recon_z0": err_z0_sum / max(n_valid_z0, 1),
        "mu_abs_mean": mu_sum / max(n_samples, 1),
    }


def fit(
    model, train_loader, val_loader, bridge: Bridge, optimizer, device,
    checkpoint_dir: Path, num_epochs: int = 300, kl_weight: float = 10.0,
    kl_warmup_epochs: int = 0, free_bits: float = 0.0, grad_clip_norm: float = 10.0,
    checkpoint_every: int = 50, start_epoch: int = 0, history: dict | None = None,
    scaler=None, accum_steps: int = 1,
) -> dict:
    """Treina o modelo.

    Roda `num_epochs` fixas, sempre — sem early stopping, igual ao ACT
    oficial (train_bc em imitate_episodes.py roda `range(num_epochs)`
    inteiro, sem parada antecipada).

    `kl_warmup_epochs > 0` ativa annealing linear do kl_weight (0 ->
    kl_weight, ao longo dessas épocas -- ver loss.kl_weight_schedule).
    Técnica de fora do ACT/paper de referência, usada para incentivar o
    decoder a de fato USAR z. NOTA: com warmup ativo, val_loss deixa de ser
    comparável entre épocas (o peso do KL muda) — a seleção por recon_z0
    fica imune, mas não tire conclusões da curva de val_loss nesse regime.

    `scaler`: passe o GradScaler de fora quando for restaurar o estado dele
    de um checkpoint (resume); com None, um novo é criado aqui.

    Com `val_loader` (comportamento original): seleção de checkpoints por
    `val_recon_z0` num top-3 VERDADEIRO (entra quem for melhor que o pior
    membro atual). Nunca interrompe o treino por falta de melhora.

    Com `val_loader=None` (Fase 2, dataset pequeno demais pra val confiável):
    treina com TODOS os dados por `num_epochs` fixo, salvando um checkpoint
    periódico a cada `checkpoint_every` épocas. A avaliação de qual desses
    checkpoints é o melhor fica pro rollout real, depois do treino.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    has_val = val_loader is not None

    if history is None:
        history = {}
    # setdefault (não dict fixo): robusto tanto a rodar sem val quanto a
    # retomar de um checkpoint salvo por uma versão anterior deste arquivo,
    # cujo history pode não ter todas as chaves de hoje (ex: mu_abs_mean).
    best_metric = min(history.get("val_recon_z0", []), default=float("inf"))
    # Resume: bests já no disco voltam a participar do ranking e da poda.
    top_k_checkpoints: list = scan_top_k(checkpoint_dir, metric_name="z0", k=TOP_K)
    if top_k_checkpoints:
        print(f"top-{TOP_K} reconstruído do disco: "
              f"{[f'{v:.4f}' for v, _ in top_k_checkpoints]}")

    for epoch in range(start_epoch, num_epochs):
        current_kl_weight = kl_weight_schedule(epoch, kl_weight, kl_warmup_epochs)
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, bridge, optimizer, scaler, device,
            current_kl_weight, free_bits, grad_clip_norm, accum_steps=accum_steps,
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

        save_checkpoint(
            checkpoint_dir / "last_checkpoint.pt", epoch, model, optimizer,
            history, scaler=scaler,
        )

        if has_val:
            # Top-k VERDADEIRO: entra quem supera o pior membro atual (ou
            # enquanto o ranking está incompleto), não só recordes globais.
            enters_top_k = (
                len(top_k_checkpoints) < TOP_K
                or va["recon_z0"] < top_k_checkpoints[-1][0]
            )
            if enters_top_k:
                top_k_checkpoints = save_top_k_checkpoint(
                    checkpoint_dir, epoch, va["recon_z0"], model,
                    history, top_k_checkpoints, k=TOP_K, metric_name="z0",
                )
                is_best = va["recon_z0"] < best_metric
                if is_best:
                    best_metric = va["recon_z0"]
                marker = "novo melhor" if is_best else "entra no top-k"
                print(f"  -> {marker} val_recon_z0: {va['recon_z0']:.4f} | "
                      f"top-{TOP_K}: {[f'{v:.4f}' for v, _ in top_k_checkpoints]}")

        if (epoch + 1) % checkpoint_every == 0:
            fname = checkpoint_dir / f"periodic_epoch{epoch:03d}.pt"
            save_checkpoint(fname, epoch, model, optimizer, history, scaler=scaler)
            print(f"  -> checkpoint periódico (época {epoch + 1})")

    return history
