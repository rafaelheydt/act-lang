"""Diagnóstico: modelos de linguagem+visão produzem um sinal espacial
específico da instrução, no domínio do LIBERO?

Antes de integrar um VLM congelado como "gate" espacial de linguagem (uma 4a
abordagem de fusão, sem nenhum parâmetro novo treinável, além de
Token/FiLM/CrossAttention em src/act_lang/models/fusion/), este script
verifica com dados reais do LIBERO se o mapa de relevância imagem-texto de
cada método é (a) espacialmente concentrado no objeto/ação certo e (b) MUDA
quando a instrução muda.

Compara CINCO métodos por padrão:
  1. CLIP -- similaridade de cosseno patch-a-patch, mesmo esquema do SigLIP
       abaixo. Baseline de contraste: isola se o ruído visto no SigLIP é
       específico dele ou fundamental de qualquer encoder só-contrastivo
       (o CLIPSeg, item 4, é construído sobre CLIP, não SigLIP).
  2-3. SigLIP 1 e SigLIP 2 -- mesmo esquema. Treino contrastivo sobre um
       vetor GLOBAL (pooled), sem supervisão espacial -- literatura de
       dense-prediction zero-shot (MaskCLIP, GEM) mostra que costuma ser
       ruidoso sem ajustes extras. SigLIP 2 (arXiv 2502.14786) foi treinado
       especificamente pra melhorar isso, mesmo porte de checkpoint.
  4. CLIPSeg (Lüddecke & Ecker, CVPR 2022, arXiv 2112.10003) -- decoder leve
       treinado sobre CLIP congelado especificamente para segmentação a
       partir de texto; já produz um mapa contínuo direto (sem produto
       escalar manual).
  5. Grounded SAM = Grounding DINO (texto -> caixa, arXiv 2303.05499) + SAM
       (caixa -> máscara de pixel) -- o mais preciso, treinado com
       supervisão de localização de verdade, mas o mais pesado (dois
       modelos encadeados) e o mais distante da ideia original de "gate
       zero-parâmetro barato".

Achado do primeiro run (SigLIP1/SigLIP2/CLIPSeg, 6 amostras, cena do
LIBERO): SigLIP 1 e 2 confirmaram o risco (diff certa-errada ~0, sinal
ruído genérico, sem vantagem clara do SigLIP 2 sobre o 1 nessa amostra).
CLIPSeg localiza bem quando os objetos comparados são categorias diferentes
(ex. lata de sopa vs. gaveta), mas confunde instruções sobre o MESMO objeto
físico (ex. "gaveta de cima" vs. "gaveta do meio" do mesmo armário) -- diff
médio negativo em 4/6 amostras, sugerindo que reconhece categoria de objeto
mas não resolve relação espacial/ordinal fina.

Imagens de câmera de robô em simulação são fora da distribuição de todos
esses modelos (treinados em fotos/dados web) -- por isso a comparação, em
vez de confiar de olho em qual "deveria" funcionar melhor.

Puramente exploratório -- não toca em src/act_lang/models nem fusion/, não
faz parte do pipeline de treino.

Uso:
    pip install -e ".[vlm]"
    python scripts/diagnose_siglip_patch_alignment.py --preprocessed-dir <caminho> --n-samples 6
    python scripts/diagnose_siglip_patch_alignment.py --hdf5-dir <caminho> --n-samples 6

    # validar só a parte mais incerta (Grounded SAM) antes de rodar tudo:
    python scripts/diagnose_siglip_patch_alignment.py --hdf5-dir <caminho> \
        --n-samples 1 --skip-siglip --skip-clipseg

Pro teste de discriminabilidade (instrução certa vs. errada) fazer sentido,
os dados precisam ter mais de uma instrução distinta -- aponte pra uma config
multi-tarefa (ex.: libero_object_language / libero_40tasks_language), não
libero_single_task.

NOTA: não testei este script ao vivo (sem dataset local nesta máquina) --
revisei com cuidado, mas rode primeiro com --n-samples 1 antes de confiar
nos heatmaps. Pontos de maior incerteza:
  - APIs internas do SiglipModel (visual_projection/text_projection) variam
    um pouco entre versões do `transformers` -- ver `_maybe_project`.
  - A assinatura exata de `processor.post_process_grounded_object_detection`
    (nomes dos kwargs de threshold, `target_sizes`) e de
    `SamProcessor`/`post_process_masks` (formato de `input_boxes`,
    `original_sizes`/`reshaped_input_sizes`) -- é o motivo do modo
    "--skip-siglip --skip-clipseg --n-samples 1" acima existir: valide essa
    parte isolada primeiro.
  - O nome exato do checkpoint SigLIP 2 base no Hub não foi confirmado
    literalmente (só "google/siglip2-so400m-patch14-384" foi visto
    confirmado) -- se "google/siglip2-base-patch16-224" não existir, o
    script avisa e segue só com os scorers que carregaram.
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

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch16"
DEFAULT_SIGLIP_MODELS = [
    "google/siglip-base-patch16-224",
    "google/siglip2-base-patch16-224",
]
DEFAULT_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM_MODEL = "facebook/sam-vit-base"


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


def resize_heatmap(heatmap: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Leva um heatmap (H',W') de qualquer método pra resolução da imagem
    original -- ponto único de resize, já que cada scorer produz numa
    resolução nativa diferente (grid de patches, saída fixa do CLIPSeg,
    máscara do SAM já na resolução original)."""
    if heatmap.shape == tuple(image_hw):
        return heatmap
    t = torch.from_numpy(heatmap).float().view(1, 1, *heatmap.shape)
    up = F.interpolate(t, size=image_hw, mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


# ---------------------------------------------------------------------------
# Scorers -- cada um implementa .name e .score(pil_image, text) -> (H',W')
# ---------------------------------------------------------------------------

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


class SiglipScorer:
    def __init__(self, model_name: str, device):
        from transformers import SiglipModel, SiglipProcessor

        self.name = model_name
        self.device = device
        self.model = SiglipModel.from_pretrained(model_name).to(device).eval()
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.grid = self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size
        self._text_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        if text not in self._text_cache:
            inputs = self.processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)
            pooled = self.model.text_model(**inputs).pooler_output
            embed = _maybe_project(getattr(self.model, "text_projection", None), pooled)
            self._text_cache[text] = F.normalize(embed, dim=-1)[0]
        return self._text_cache[text]

    @torch.no_grad()
    def score(self, pil_image: Image.Image, text: str) -> np.ndarray:
        inputs = self.processor(images=[pil_image], return_tensors="pt").to(self.device)
        patch_hidden = self.model.vision_model(**inputs).last_hidden_state  # (1, n_patches, hidden) -- SigLIP não usa token CLS
        patch_embed = _maybe_project(getattr(self.model, "visual_projection", None), patch_hidden)
        patch_embed = F.normalize(patch_embed, dim=-1)[0]  # (n_patches, proj_dim)

        text_embed = self._encode_text(text)
        sim = patch_embed @ text_embed  # (n_patches,)
        assert self.grid * self.grid == sim.shape[0], (
            f"{self.name}: grid {self.grid}x{self.grid} não bate com {sim.shape[0]} patches"
        )
        return sim.view(self.grid, self.grid).cpu().numpy()


class ClipScorer:
    """Mesmo esquema do SiglipScorer (produto escalar patch-a-patch), mas
    com CLIP -- baseline de contraste pra isolar se o ruído visto no SigLIP
    é específico dele ou fundamental de qualquer encoder só-contrastivo.
    Duas diferenças em relação ao SigLIP: o ViT do CLIP tem um token CLS
    prependado em last_hidden_state (descartado abaixo antes do reshape), e
    o tokenizer de texto usa padding dinâmico normal (sem o gotcha
    padding="max_length" que o SigLIP exige)."""

    def __init__(self, model_name: str, device):
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_name
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.grid = self.model.config.vision_config.image_size // self.model.config.vision_config.patch_size
        self._text_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        if text not in self._text_cache:
            inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
            pooled = self.model.text_model(**inputs).pooler_output
            embed = _maybe_project(getattr(self.model, "text_projection", None), pooled)
            self._text_cache[text] = F.normalize(embed, dim=-1)[0]
        return self._text_cache[text]

    @torch.no_grad()
    def score(self, pil_image: Image.Image, text: str) -> np.ndarray:
        inputs = self.processor(images=[pil_image], return_tensors="pt").to(self.device)
        patch_hidden = self.model.vision_model(**inputs).last_hidden_state[:, 1:, :]  # descarta o token CLS (posição 0)
        patch_embed = _maybe_project(getattr(self.model, "visual_projection", None), patch_hidden)
        patch_embed = F.normalize(patch_embed, dim=-1)[0]  # (n_patches, proj_dim)

        text_embed = self._encode_text(text)
        sim = patch_embed @ text_embed  # (n_patches,)
        assert self.grid * self.grid == sim.shape[0], (
            f"{self.name}: grid {self.grid}x{self.grid} não bate com {sim.shape[0]} patches (após descartar CLS)"
        )
        return sim.view(self.grid, self.grid).cpu().numpy()


class ClipSegScorer:
    def __init__(self, model_name: str, device):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.name = model_name
        self.device = device
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPSegProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score(self, pil_image: Image.Image, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**inputs).logits  # forma varia por versão: (1,H',W') ou (H',W')
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits[0])  # treino com BCE -- sigmoid vira probabilidade por posição
        return probs.cpu().numpy()


class GroundedSamScorer:
    """Grounding DINO (texto -> caixas) + SAM (caixa -> máscara). Ver aviso
    de incerteza de API no docstring do módulo -- valide isolado primeiro
    (--skip-siglip --skip-clipseg --n-samples 1)."""

    def __init__(self, dino_model: str, sam_model: str, device, box_threshold: float, text_threshold: float):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection, SamModel, SamProcessor

        self.name = f"grounded-sam ({dino_model.split('/')[-1]}+{sam_model.split('/')[-1]})"
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.dino_processor = AutoProcessor.from_pretrained(dino_model)
        self.dino_model = GroundingDinoForObjectDetection.from_pretrained(dino_model).to(device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_model)
        self.sam_model = SamModel.from_pretrained(sam_model).to(device).eval()

    @staticmethod
    def _format_query(text: str) -> str:
        # convenção de treino do Grounding DINO: minúsculo, terminado em ponto
        q = text.lower().strip()
        return q if q.endswith(".") else q + "."

    @torch.no_grad()
    def score(self, pil_image: Image.Image, text: str) -> np.ndarray:
        w, h = pil_image.size
        query = self._format_query(text)

        dino_inputs = self.dino_processor(images=pil_image, text=query, return_tensors="pt").to(self.device)
        dino_outputs = self.dino_model(**dino_inputs)
        results = self.dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            input_ids=dino_inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]
        boxes = results["boxes"]  # (N,4) xyxy em pixels da imagem original
        scores = results["scores"]

        if boxes is None or len(boxes) == 0:
            # nenhuma detecção -- resultado válido: um bom detector não deveria
            # "achar" nada pra uma instrução que não bate com a imagem
            return np.zeros((h, w), dtype=np.float32)

        sam_inputs = self.sam_processor(pil_image, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        sam_outputs = self.sam_model(**sam_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (n_boxes, n_masks_por_caixa, H, W) -- pega a melhor máscara por caixa via iou_scores

        best_mask_idx = sam_outputs.iou_scores[0].argmax(dim=-1).cpu()  # (n_boxes,)
        heatmap = np.zeros((h, w), dtype=np.float32)
        for i, box_score in enumerate(scores.tolist()):
            mask_i = masks[i][best_mask_idx[i]].numpy().astype(np.float32)
            heatmap = np.maximum(heatmap, mask_i * box_score)
        return heatmap


def build_scorers(args, device) -> list:
    scorers = []

    if not args.skip_clip:
        print(f"carregando {args.clip_model}...")
        try:
            scorers.append(ClipScorer(args.clip_model, device))
        except Exception as e:
            print(f"  AVISO: falha ao carregar CLIP '{args.clip_model}' ({e}) -- pulando. Confira o id exato "
                  f"em https://huggingface.co/models?search={args.clip_model.split('/')[-1]}")

    if not args.skip_siglip:
        for name in args.siglip_models:
            print(f"carregando {name}...")
            try:
                scorers.append(SiglipScorer(name, device))
            except Exception as e:
                print(f"  AVISO: falha ao carregar '{name}' ({e}) -- pulando. Confira o id exato "
                      f"em https://huggingface.co/models?search={name.split('/')[-1]}")

    if not args.skip_clipseg:
        print(f"carregando {args.clipseg_model}...")
        try:
            scorers.append(ClipSegScorer(args.clipseg_model, device))
        except Exception as e:
            print(f"  AVISO: falha ao carregar CLIPSeg '{args.clipseg_model}' ({e}) -- pulando.")

    if not args.skip_grounded_sam:
        print(f"carregando Grounded SAM ({args.dino_model} + {args.sam_model})...")
        try:
            scorers.append(GroundedSamScorer(
                args.dino_model, args.sam_model, device, args.box_threshold, args.text_threshold
            ))
        except Exception as e:
            print(f"  AVISO: falha ao carregar Grounded SAM ({e}) -- pulando. Rode primeiro com "
                  f"--n-samples 1 --skip-siglip --skip-clipseg pra isolar o problema.")

    assert scorers, "nenhum scorer carregado com sucesso -- confira os argumentos --*-model(s)/--skip-*"
    return scorers


def save_comparison(image_np, per_scorer_results: list[tuple], task_correct: str, task_wrong: str,
                     out_path: Path) -> None:
    """per_scorer_results: lista de (nome, heatmap_correct, heatmap_wrong,
    sim_max_correct, sim_max_wrong) -- uma linha do grid por scorer."""
    n = len(per_scorer_results)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n), squeeze=False)

    for row, (name, heatmap_correct, heatmap_wrong, sim_max_correct, sim_max_wrong) in enumerate(
        per_scorer_results
    ):
        axes[row][0].imshow(image_np)
        axes[row][0].set_title(f"{name}\nimagem original", fontsize=8)
        axes[row][0].axis("off")

        panels = [
            (axes[row][1], heatmap_correct, task_correct, sim_max_correct, "instrução CERTA"),
            (axes[row][2], heatmap_wrong, task_wrong, sim_max_wrong, "instrução ERRADA"),
        ]
        for ax, heatmap, task, sim_max, label in panels:
            ax.imshow(image_np)
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            ax.set_title(f"{label}\n\"{task[:40]}\"\nmax={sim_max:.3f}", fontsize=8)
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
    parser.add_argument("--output-dir", type=Path, default=Path("siglip_patch_alignment_outputs"))
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Seed pra escolha da 'instrução errada'.")

    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL,
                         help="Checkpoint CLIP -- baseline de contraste com o esquema do SigLIP.")
    parser.add_argument("--siglip-models", nargs="+", default=DEFAULT_SIGLIP_MODELS,
                         help="Checkpoints SigLIP a comparar. Default: SigLIP 1 base vs. SigLIP 2 base.")
    parser.add_argument("--clipseg-model", default=DEFAULT_CLIPSEG_MODEL)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)

    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-siglip", action="store_true")
    parser.add_argument("--skip-clipseg", action="store_true")
    parser.add_argument("--skip-grounded-sam", action="store_true")
    args = parser.parse_args()

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

    scorers = build_scorers(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diffs_by_scorer: dict[str, list[float]] = {s.name: [] for s in scorers}
    for i, sample in enumerate(samples):
        image_tensor = sample["observation.images.image"]  # (3,H,W) float32 [0,1]
        image_np = image_tensor.permute(1, 2, 0).numpy()
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        image_hw = image_np.shape[:2]

        per_scorer_results = []
        summary_parts = []
        for scorer in scorers:
            try:
                raw_correct = scorer.score(pil_image, tasks_correct[i])
                raw_wrong = scorer.score(pil_image, tasks_wrong[i])
            except Exception as e:
                print(f"  AVISO: {scorer.name} falhou na amostra {i} ({e}) -- pulando esta linha.")
                continue
            heatmap_correct = resize_heatmap(raw_correct, image_hw)
            heatmap_wrong = resize_heatmap(raw_wrong, image_hw)

            sim_max_correct = float(heatmap_correct.max())
            sim_max_wrong = float(heatmap_wrong.max())
            diff = sim_max_correct - sim_max_wrong
            diffs_by_scorer[scorer.name].append(diff)

            per_scorer_results.append((scorer.name, heatmap_correct, heatmap_wrong, sim_max_correct, sim_max_wrong))
            summary_parts.append(f"{scorer.name}: diff={diff:+.3f}")

        if not per_scorer_results:
            print(f"[{i}] todos os scorers falharam nesta amostra -- pulando PNG.")
            continue

        out_path = args.output_dir / f"sample_{i:02d}.png"
        save_comparison(image_np, per_scorer_results, tasks_correct[i], tasks_wrong[i], out_path)
        print(f"[{i}] certa=\"{tasks_correct[i][:50]}\" | errada=\"{tasks_wrong[i][:50]}\" | "
              f"{' | '.join(summary_parts)} -> {out_path}")

    print(f"\n=== diff médio (max certa - max errada) nas {len(samples)} amostras, por scorer ===")
    for name, diffs in diffs_by_scorer.items():
        if diffs:
            print(f"  {name}: {sum(diffs) / len(diffs):+.4f}")
        else:
            print(f"  {name}: sem amostras válidas")
    print()
    print("Como interpretar: diff consistentemente positivo e claramente > 0 é sinal de que o mapa")
    print("de relevância é específico da instrução -- o gate teria informação real pra explorar.")
    print("diff perto de zero, ou heatmaps visualmente quase idênticos entre instrução certa/errada")
    print("nos PNGs salvos, indica saliência genérica, não alinhamento com a linguagem. O scorer com")
    print("diff claramente maior é o melhor candidato pra uma eventual integração -- mas se for o")
    print("Grounded SAM, lembre que ele não é mais 'zero-parâmetro barato': é dois modelos grandes")
    print("encadeados, então a decisão vira 'vale o custo de engenharia?', não só 'existe sinal?'.")


if __name__ == "__main__":
    main()
