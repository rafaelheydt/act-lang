"""Lê o histórico de losses salvo num checkpoint e mostra/plota a evolução.

O `history` (train_loss, val_recon_z0, etc.) é salvo DENTRO do checkpoint a
cada época (ver training/checkpoints.py) -- então dá pra ver a curva
completa mesmo que o log do terminal esteja bufferizado/atrasado.

Uso:
    python scripts/plot_history.py /caminho/pro/last_checkpoint.pt
    python scripts/plot_history.py /caminho/pro/last_checkpoint.pt --save curves.png

Nota: prefira ler um `best_epochNNN_*.pt` (escrito uma vez, nunca mais
tocado) em vez de `last_checkpoint.pt` se o treino ainda estiver rodando --
ler o `last_checkpoint.pt` bem no instante em que o processo de treino está
regravando ele (torch.save) é uma pequena janela de corrida (race condition)
que pode dar erro de leitura. Se acontecer, é só tentar de novo.
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Caminho pro .pt (ex: last_checkpoint.pt)")
    parser.add_argument("--save", type=Path, default=None, help="Salva um PNG das curvas nesse caminho")
    parser.add_argument("--no-table", action="store_true", help="Não imprime a tabela no terminal")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    history = ckpt["history"]
    last_epoch = ckpt["epoch"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Última época salva neste arquivo: {last_epoch}")
    print()

    keys = list(history.keys())
    n_epochs = len(next(iter(history.values()))) if history else 0

    if not args.no_table and n_epochs:
        header = "epoch".ljust(7) + "".join(k.ljust(16) for k in keys)
        print(header)
        for i in range(n_epochs):
            row = str(i).ljust(7)
            for k in keys:
                v = history[k][i] if i < len(history[k]) else float("nan")
                row += f"{v:.4f}".ljust(16)
            print(row)

    if args.save:
        if n_epochs == 0:
            print("\nhistory vazio -- nada para plotar.")
            return
        import matplotlib
        matplotlib.use("Agg")  # sem display -- salva direto em arquivo
        import matplotlib.pyplot as plt

        metrics = sorted({k.split("_", 1)[1] for k in history})
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            train_key, val_key = f"train_{metric}", f"val_{metric}"
            if train_key in history:
                ax.plot(history[train_key], label="train")
            if val_key in history:
                ax.plot(history[val_key], label="val")
            ax.set_title(metric)
            ax.set_xlabel("epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.save, dpi=120)
        print(f"\nCurvas salvas em: {args.save}")


if __name__ == "__main__":
    main()
