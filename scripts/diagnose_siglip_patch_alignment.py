"""Diagnóstico: o SigLIP produz um sinal patch-a-patch específico da instrução?

Antes de integrar SigLIP congelado como um "gate" espacial de linguagem (uma
4a abordagem de fusão, sem nenhum parâmetro novo treinável, além de
Token/FiLM/CrossAttention em src/act_lang/models/fusion/), este script
verifica com dados reais do LIBERO se a similaridade patch-a-patch entre
imagem e texto do SigLIP é (a) espacialmente concentrada no objeto/ação
certo e (b) MUDA quando a instrução muda.

O SigLIP é treinado com uma loss contrastiva sobre um vetor GLOBAL (pooled)
-- não há garantia de que a similaridade por patch seja discriminativa.
Literatura de dense-prediction zero-shot com CLIP/SigLIP (MaskCLIP, GEM)
mostra que costuma ser ruidosa sem ajustes extras, e imagens de câmera de
robô em simulação são bem fora da distribuição de fotos web que o SigLIP viu
no pré-treino.

Puramente exploratório -- não toca em src/act_lang/models nem fusion/, não
faz parte do pipeline de treino.

Uso:
    pip install -e ".[vlm]"
    python scripts/diagnose_siglip_patch_alignment.py --preprocessed-dir <caminho> --n-samples 6
    python scripts/diagnose_siglip_patch_alignment.py --hdf5-dir <caminho> --n-samples 6

Pro teste de discriminabilidade (instrução certa vs. errada) fazer sentido,
os dados precisam ter mais de uma instrução distinta -- aponte pra uma config
multi-tarefa (ex.: libero_object_language / libero_40tasks_language), não
libero_single_task.

NOTA: não testei este script ao vivo (sem dataset local nesta máquina) --
revisei com cuidado, mas rode primeiro com --n-samples 2 pra conferir que
tudo carrega certo antes de confiar nos heatmaps. As APIs internas do
SiglipModel (visual_projection/text_projection) variam um pouco entre
versões do `transformers`; o código abaixo tolera a ausência delas (ver
`_maybe_project`).
"""

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from act_lang.utils.runtime import pick_device


def load_dataset(args):
    if args.hdf5_dir is not None:
        from act_lang.data.hdf5_libero import HDF5LiberoDataset
        return HDF5LiberoDataset(args.hdf5_dir, pred_horizon=1)
    from act_lang.data.preprocessed import PreprocessedLiberoDataset
    return PreprocessedLiberoDataset(args.preprocessed_dir, pred_horizon=1)


def sample_indices(n_total: int, n_samples: int) -> list[int]:
    """Espalha os índices ao longo do dataset (não só os primeiros N frames,
    que tenderiam a vir do mesmo episódio/instrução)."""
    if n_samples >= n_total:
        return list(range(n_total))
    step = n_total // n_samples
    return [i * step for i in range(n_samples)]


def _maybe_project(module_or_none, x: torch.Tensor) -> torch.Tensor:
    """Algumas versões do SiglipModel expõem visual_projection/text_projection
    (como o CLIPModel); outras já devolvem os embeddings no espaço
    compartilhado direto do pooling/hidden_size, sem projeção extra. Se o
    atributo não existir, usamos o tensor como está -- dimensionalmente
    válido nos dois casos porque o SigLIP configura vision.hidden_size ==
    text.hidden_size exatamente para isso."""
    if module_or_none is None:
        return x
    return module_or_none(x)


@torch.no_grad()
def encode_text(model, processor, texts: list[str], device) -> torch.Tensor:
    inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(device)
    pooled = model.text_model(**inputs).pooler_output
    embed = _maybe_project(getattr(model, "text_projection", None), pooled)
    return F.normalize(embed, dim=-1)  # (n_texts, proj_dim)


@torch.no_grad()
def encode_image_patches(model, processor, pil_image: Image.Image, device) -> torch.Tensor:
    inputs = processor(images=[pil_image], return_tensors="pt").to(device)
    patch_hidden = model.vision_model(**inputs).last_hidden_state  # (1, n_patches, hidden) -- SigLIP não usa token CLS, então todo token aqui é espacial
    patch_embed = _maybe_project(getattr(model, "visual_projection", None), patch_hidden)
    return F.normalize(patch_embed, dim=-1)[0]  # (n_patches, proj_dim)


def similarity_heatmap(patch_embeds: torch.Tensor, text_embed: torch.Tensor,
                        grid: int, image_hw: tuple[int, int]) -> np.ndarray:
    sim = patch_embeds @ text_embed  # (n_patches,)
    assert grid * grid == sim.shape[0], (
        f"grid {grid}x{grid} não bate com {sim.shape[0]} patches -- "
        "confira patch_size/image_size do checkpoint escolhido"
    )
    sim_grid = sim.view(1, 1, grid, grid)
    upsampled = F.interpolate(sim_grid, size=image_hw, mode="bilinear", align_corners=False)
    return upsampled[0, 0].cpu().numpy()


def save_comparison(image_np, heatmap_correct, heatmap_wrong, task_correct, task_wrong,
                     sim_max_correct, sim_max_wrong, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(image_np)
    axes[0].set_title("imagem original", fontsize=9)
    axes[0].axis("off")

    panels = [
        (axes[1], heatmap_correct, task_correct, sim_max_correct, "instrução CERTA"),
        (axes[2], heatmap_wrong, task_wrong, sim_max_wrong, "instrução ERRADA"),
    ]
    for ax, heatmap, task, sim_max, label in panels:
        ax.imshow(image_np)
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        ax.set_title(f"{label}\n\"{task[:40]}\"\nsim_max={sim_max:.3f}", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--preprocessed-dir", type=Path, default=None,
                      help="Pasta gerada por scripts/preprocess_dataset.py")
    src.add_argument("--hdf5-dir", type=Path, default=None,
                      help="Pasta gerada por scripts/download_libero_hdf5.py")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--model-name", default="google/siglip-base-patch16-224",
                         help="Trocar por google/siglip-so400m-patch14-384 (backbone do PaliGemma) "
                              "se o resultado do base parecer promissor.")
    parser.add_argument("--output-dir", type=Path, default=Path("siglip_patch_alignment_outputs"))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Seed pra escolha da 'instrução errada'.")
    args = parser.parse_args()

    from transformers import SiglipModel, SiglipProcessor

    random.seed(args.seed)
    device = pick_device(preferred_index=args.device_index)
    print(f"device: {device}")

    dataset = load_dataset(args)
    all_task_texts = sorted(set(dataset.episode_task_labels.values()))
    print(f"dataset: {len(dataset)} amostras, {len(dataset.episode_task_labels)} episódios, "
          f"{len(all_task_texts)} instrução(ões) distinta(s)")
    if len(all_task_texts) < 2:
        print("AVISO: só 1 instrução distinta no dataset -- o teste de discriminabilidade não é")
        print("significativo (não existe 'instrução errada' de verdade pra comparar). Aponte")
        print("--preprocessed-dir/--hdf5-dir pra uma config multi-tarefa pra esse teste valer algo.")

    indices = sample_indices(len(dataset), args.n_samples)
    samples = [dataset[i] for i in indices]
    tasks_correct = [s["task"] for s in samples]
    tasks_wrong = [
        random.choice([t for t in all_task_texts if t != task] or [task])
        for task in tasks_correct
    ]

    print(f"carregando {args.model_name}...")
    model = SiglipModel.from_pretrained(args.model_name).to(device).eval()
    processor = SiglipProcessor.from_pretrained(args.model_name)
    grid = model.config.vision_config.image_size // model.config.vision_config.patch_size

    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_embeds_correct = encode_text(model, processor, tasks_correct, device)
    text_embeds_wrong = encode_text(model, processor, tasks_wrong, device)

    diffs = []
    for i, sample in enumerate(samples):
        image_tensor = sample["observation.images.image"]  # (3,H,W) float32 [0,1]
        image_np = image_tensor.permute(1, 2, 0).numpy()
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))

        patch_embeds = encode_image_patches(model, processor, pil_image, device)
        image_hw = image_np.shape[:2]
        heatmap_correct = similarity_heatmap(patch_embeds, text_embeds_correct[i], grid, image_hw)
        heatmap_wrong = similarity_heatmap(patch_embeds, text_embeds_wrong[i], grid, image_hw)

        sim_max_correct = float(heatmap_correct.max())
        sim_max_wrong = float(heatmap_wrong.max())
        diffs.append(sim_max_correct - sim_max_wrong)

        out_path = args.output_dir / f"sample_{i:02d}.png"
        save_comparison(image_np, heatmap_correct, heatmap_wrong, tasks_correct[i], tasks_wrong[i],
                         sim_max_correct, sim_max_wrong, out_path)
        print(f"[{i}] certa=\"{tasks_correct[i][:50]}\" (sim_max={sim_max_correct:.3f}) | "
              f"errada=\"{tasks_wrong[i][:50]}\" (sim_max={sim_max_wrong:.3f}) | "
              f"diff={diffs[-1]:+.3f} -> {out_path}")

    mean_diff = sum(diffs) / len(diffs)
    print(f"\n=== diff médio (sim_max certa - sim_max errada) nas {len(samples)} amostras: {mean_diff:+.4f} ===")
    print("Como interpretar: diff consistentemente positivo e claramente > 0 nas amostras é sinal de")
    print("que a similaridade patch-a-patch é específica da instrução -- o gate teria informação real")
    print("pra explorar. diff perto de zero, ou heatmaps visualmente quase idênticos entre instrução")
    print("certa/errada nos PNGs salvos, confirma o risco: o sinal é saliência genérica, não")
    print("alinhamento com a linguagem, e o gate zero-parâmetro provavelmente não vale a pena sem")
    print("alguma forma de calibração/fine-tuning.")


if __name__ == "__main__":
    main()
