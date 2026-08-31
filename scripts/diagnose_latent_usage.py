"""Diagnóstico: o decoder está usando z de verdade, ou ignorando?

O log de treino já compara z=mu vs z=0 (dois pontos fixos) -- este script
vai além, amostrando N valores de z ~ N(0,1) (o prior de verdade) pra cada
exemplo real, e mede o quanto a predição de ações VARIA entre amostras.
Se a variação for pequena perto da escala normal do erro de reconstrução,
é sinal de que o decoder não está usando z de forma significativa.

Uso:
    python scripts/diagnose_latent_usage.py --config language_40_film

NOTA: não testei este script ao vivo (problema de ambiente do meu lado) --
revisei com cuidado, mas rode primeiro com --n-batches 1 pra conferir que
tudo carrega certo antes de confiar nos números.
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import importlib

import torch

from act_lang.models.act import ACT
from act_lang.models.fusion import build_fusion
from act_lang.training.loss import masked_l1
from act_lang.utils.runtime import pick_device

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


def build_model(cfg: dict, device: torch.device) -> ACT:
    fusion = build_fusion(cfg.get("fusion_type"), d_model=cfg["d_model"])
    model = ACT(
        action_dim=cfg["action_dim"], state_dim=cfg["state_dim"],
        d_model=cfg["d_model"], latent_dim=cfg["latent_dim"],
        chunk_size=cfg["chunk_size"], n_cameras=cfg["n_cameras"],
        n_encoder_layers=cfg["n_encoder_layers"], n_decoder_layers=cfg["n_decoder_layers"],
        n_heads=cfg["n_heads"], dropout=cfg["dropout"], pretrained_backbone=False,
        decoder_style=cfg["decoder_style"], fusion=fusion,
    ).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=sorted(CONFIG_REGISTRY))
    parser.add_argument("--checkpoint", default=None, help="Caminho do .pt (padrão: last_checkpoint.pt do experiment_name)")
    parser.add_argument("--n-samples", type=int, default=20, help="Quantas amostras de z ~ N(0,1) testar por exemplo")
    parser.add_argument("--n-batches", type=int, default=3, help="Quantos batches de validação examinar")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(preferred_index=cfg.get("device_index"))
    print(f"config: {args.config} | device: {device}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        Path.home() / "act-lang-checkpoints" / cfg["experiment_name"] / "last_checkpoint.pt"
    )
    print(f"checkpoint: {ckpt_path}")

    model = build_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"checkpoint da época {ckpt['epoch']} carregado\n")

    # reaproveita o pipeline de dados real do próprio treino
    from scripts.train import build_data
    _, val_loader, bridge = build_data(cfg, device)

    torch.manual_seed(0)
    all_spreads = []

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= args.n_batches:
                break
            images, state, actions, is_pad, task_texts = bridge(batch, device)
            # só o 1o exemplo do batch, pra manter o print legível
            images, state, actions = images[:1], state[:1], actions[:1]
            is_pad = is_pad[:1] if is_pad is not None else None
            task_texts = task_texts[:1] if task_texts is not None else None

            # referências: mu (posterior) e z=0 (prior fixo) -- mesmos dois
            # pontos que o log de treino já mostra, aqui recalculados pra
            # comparação lado a lado com as amostras do prior de verdade
            mu, logvar = model.cvae_encoder(state, actions, is_pad)
            a_hat_mu = model.decode_with_z(images, state, mu, task_texts)
            z0 = torch.zeros(1, model.latent_dim, device=device)
            a_hat_z0 = model.decode_with_z(images, state, z0, task_texts)
            diff_mu_z0 = (a_hat_mu - a_hat_z0).abs().mean().item()

            # N amostras do PRIOR de verdade, z ~ N(0,1) -- não o posterior
            samples = torch.randn(args.n_samples, model.latent_dim, device=device)
            preds = []
            for i in range(args.n_samples):
                z_i = samples[i : i + 1]
                a_hat_i = model.decode_with_z(images, state, z_i, task_texts)
                preds.append(a_hat_i)
            preds = torch.cat(preds, dim=0)  # (n_samples, chunk, action_dim)

            spread = preds.std(dim=0).mean().item()  # std entre amostras, média sobre (chunk, action_dim)
            recon_error_scale = masked_l1(a_hat_mu, actions, is_pad).item()  # escala de referência

            all_spreads.append(spread)
            print(f"--- exemplo {b_idx} ---")
            print(f"  |a_hat(mu) - a_hat(z=0)|.mean() = {diff_mu_z0:.6f}")
            print(f"  std entre {args.n_samples} amostras de z~N(0,1)   = {spread:.6f}")
            print(f"  erro de reconstrução (mu vs ação real), p/ escala = {recon_error_scale:.6f}")
            print(f"  razão spread/recon_error = {spread / max(recon_error_scale, 1e-8):.4f}")
            print()

    media = sum(all_spreads) / len(all_spreads) if all_spreads else float("nan")
    print(f"=== spread médio entre amostras do prior, nos {len(all_spreads)} exemplos: {media:.6f} ===")
    print()
    print("Como interpretar: se 'razão spread/recon_error' for << 1 (ex: < 0.1)")
    print("em todos os exemplos, o decoder está essencialmente ignorando z --")
    print("variar z quase não muda a predição, comparado ao tamanho normal do")
    print("erro de reconstrução. Se for próximo de 1 ou maior, z está tendo")
    print("efeito real na predição.")


if __name__ == "__main__":
    main()
