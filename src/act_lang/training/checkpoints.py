"""Salvamento/retomada de checkpoints e ranking top-k.

CORREÇÕES aplicadas:
  - `load_checkpoint` retorna `epoch + 1` como start_epoch — o notebook
    original re-executava a época já salva ao retomar (off-by-one).
  - Escrita ATÔMICA (tmp + os.replace): se a sessão do Colab morrer no meio
    do torch.save do last_checkpoint.pt, o arquivo anterior sobrevive
    intacto e o --resume continua funcionando. Sem isso, a interrupção
    corrompe exatamente o arquivo que existe pra proteger contra interrupção.
  - Estado do GradScaler entra no checkpoint (e é restaurado no resume):
    sem ele, cada retomada zera o scale pro valor inicial e re-faz o warmup
    do AMP — steps pulados por overflow logo após cada resume.
  - Checkpoints "best" salvam SÓ o modelo (optimizer=None): o estado do
    AdamW (~2 buffers por parâmetro) triplica o tamanho do arquivo e só é
    necessário pra retomar treino — o que se faz pelo last_checkpoint.pt,
    nunca por um best.
  - `scan_top_k` reconstrói o ranking a partir dos arquivos best_*.pt já
    no disco — no resume, os bests de antes da interrupção voltam a
    participar da poda (antes, a lista nascia vazia e os arquivos antigos
    acumulavam pra sempre).
"""

import os
import re
from pathlib import Path

import torch


def save_checkpoint(
    path: Path, epoch: int, model, optimizer=None, history: dict | None = None,
    scaler=None,
) -> None:
    """Escrita atômica: serializa em `path.tmp` e move por cima com
    os.replace (rename atômico no mesmo filesystem)."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history if history is not None else {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    tmp_path = Path(str(path) + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path, model, optimizer=None, device="cpu", scaler=None
) -> tuple[int, dict]:
    """Retorna (start_epoch, history) — start_epoch é a PRÓXIMA época a rodar.

    `optimizer`/`scaler` são restaurados se passados E presentes no arquivo
    (checkpoints antigos, de antes do scaler entrar no payload, seguem
    carregáveis — o scaler só fica sem restaurar, com um aviso).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None:
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        else:
            print("aviso: checkpoint sem estado do GradScaler (formato antigo) "
                  "-- scaler reinicia do zero, warmup do AMP se repete.")
    return ckpt["epoch"] + 1, ckpt["history"]


def scan_top_k(checkpoint_dir: Path, metric_name: str = "z0", k: int = 3) -> list:
    """Reconstrói a lista top-k [(valor, Path), ...] a partir dos arquivos
    best_*.pt existentes em `checkpoint_dir` (para resume). Arquivos além
    dos k melhores são podados na hora, mantendo o disco consistente.

    A regex é construída COM o metric_name (não genérica): no formato de
    nome `best_epoch{e:03d}_{metric_name}{valor:.4f}.pt` não há separador
    entre nome e valor, e "z0" termina em dígito — um split genérico
    engoliria o "0" do nome como parte do valor. Ancorar no nome conhecido
    é inambíguo e lê também os arquivos de runs antigas (mesmo formato).
    """
    best_re = re.compile(
        rf"best_epoch(\d+)_{re.escape(metric_name)}([\d.]+)\.pt$"
    )
    found = []
    for f in sorted(Path(checkpoint_dir).glob("best_epoch*.pt")):
        m = best_re.search(f.name)
        if m:
            found.append((float(m.group(2)), f))
    found.sort(key=lambda x: x[0])
    for _, extra in found[k:]:  # poda o excedente herdado de runs anteriores
        extra.unlink(missing_ok=True)
    return found[:k]


def save_top_k_checkpoint(
    checkpoint_dir: Path, epoch: int, metric_value: float, model,
    history: dict, top_k_list: list, k: int = 3, metric_name: str = "z0",
) -> list:
    """Insere este checkpoint no ranking e poda além de k. A DECISÃO de
    salvar (métrica melhor que o pior do top-k atual, ou lista incompleta)
    é de quem chama — ver fit(). Best NÃO carrega optimizer (ver módulo)."""
    fname = checkpoint_dir / f"best_epoch{epoch:03d}_{metric_name}{metric_value:.4f}.pt"
    save_checkpoint(fname, epoch, model, optimizer=None, history=history)
    top_k_list.append((metric_value, fname))
    top_k_list.sort(key=lambda x: x[0])
    while len(top_k_list) > k:
        _, removed_fname = top_k_list.pop()
        if removed_fname.exists():
            removed_fname.unlink()
    return top_k_list
