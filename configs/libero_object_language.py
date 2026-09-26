"""Config: LIBERO, 10 tarefas do libero_object, COM linguagem (Fase 3).

Base idêntica à Fase 2 (mesmas 10 tarefas, mesmo split min_holdout) -- só
adiciona a fusão de linguagem. Comparar contra o "chão" da Fase 2
(experiment_name="libero_v2_10tasks_sem_lingua") é o que dá sentido
científico a esta fase: quanto a linguagem melhora em relação a não ter
instrução nenhuma? E comparar os 3 experimentos abaixo entre si é a
pergunta central da dissertação: qual mecanismo de fusão funciona melhor?

`make_config(fusion_type, backbone_type)` gera os configs a partir de uma
base compartilhada -- evita triplicar/duplicar dezenas de linhas quase
idênticas, mas cada CONFIG_* continua sendo um dict comum, do jeito que o
resto do projeto (notebooks, fit(), etc.) já espera.

`backbone_type` (ablação de backbones pareados, ver docs/hdf5_migration.md
e models/backbones/): "resnet18" (default, ResNet18+MiniLM, comportamento
original) | "clip_vitb32" (CLIP ViT-B/32, imagem+texto pareados) |
"siglip2_base" (SigLIP2-base-patch16-224, imagem+texto pareados). Os
backbones novos vêm CONGELADOS por padrão (`freeze_vision_backbone`) --
são ~8x maiores que o ResNet18 atual, pré-treinados em escala muito maior;
fine-tuning completo com o pouco dado do LIBERO arrisca destruir a
representação pré-treinada sem ganho compensador claro.
"""

from configs.libero_object_multitask import TASK_TEXTS_LIBERO_OBJECT_10

_BACKBONE_TYPES = ("resnet18", "clip_vitb32", "siglip2_base")


def make_config(fusion_type: str, backbone_type: str = "resnet18") -> dict:
    assert fusion_type in ("token", "film", "cross_attn"), fusion_type
    assert backbone_type in _BACKBONE_TYPES, backbone_type
    name_suffix = fusion_type if backbone_type == "resnet18" else f"{backbone_type}_{fusion_type}"
    return {
        "experiment_name": f"libero_v2_10tasks_lingua_{name_suffix}",
        "device_index": None,  # None = auto (GPU com mais memória livre)
        "backbone_type": backbone_type,
        "freeze_vision_backbone": backbone_type != "resnet18",
        # dados -- MESMAS 10 tarefas e split da Fase 2, pra comparação direta
        "task_texts": TASK_TEXTS_LIBERO_OBJECT_10,
        "task_suite_name": "libero_object",
        "val_frac": 0.1,  # usado só se val_strategy="fraction"
        "val_strategy": "min_holdout",
        "n_val_per_task": 1,
        "seed": 42,
        "obs_horizon": 1,
        "pred_horizon": 50,
        "batch_size": 32,
        # modelo
        "action_dim": 7,
        "state_dim": 8,
        "d_model": 512,
        "latent_dim": 32,
        "chunk_size": 50,
        "n_cameras": 2,
        "n_encoder_layers": 4,
        "n_decoder_layers": 4,
        "n_heads": 8,
        "dropout": 0.1,
        "freeze_bn": True,
        "decoder_style": "detr",
        "fusion_type": fusion_type,  # "token" | "film" | "cross_attn" -> build_fusion()
        # otimização -- idêntico à Fase 2, único fator que muda é fusion_type
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "num_epochs": 300,
        "kl_weight": 10.0,
        "kl_warmup_epochs": 10,  # annealing linear 0->kl_weight -- mesma correção de
        # libero_40tasks_language.py (diagnóstico de 31/08, decoder ignorando z):
        # sem isso, kl_weight=10 em força total desde a época 1 colapsa o
        # posterior rápido (visto ao vivo: |mu| e kld indo a ~0 em 4 épocas
        # com backbone_type="clip_vitb32").
        "free_bits": 0.05,  # reativado -- mesma correção (diagnóstico de colapso de
        # posterior em 30/08) -- 0.0 (valor anterior) não dava nenhuma folga
        # de KL por dimensão, incentivando o otimizador a zerar z rápido demais.
        "grad_clip_norm": 10.0,
        "checkpoint_every": 50,
        # rollout -- mesma ressalva da Fase 2: com 10 tarefas, cada uma
        # precisa de env/task_id próprio (notebook de rollout ainda não
        # generalizado pra multi-task).
        "rollout_m": 0.01,
        "rollout_max_steps": 300,
        "rollout_n_episodes": 10,
    }


CONFIG_TOKEN = make_config("token")          # ResNet18 original -- inalterado
CONFIG_FILM = make_config("film")
CONFIG_CROSS_ATTN = make_config("cross_attn")

# Ablação leve: os 2 backbones novos x os 3 mecanismos de fusão já
# existentes -- FiLM é a prioridade (CONFIG_CLIP_FILM/CONFIG_SIGLIP_FILM),
# Token/CrossAttention continuam disponíveis sem precisar de código
# especial por combinação.
_ABLATION_BACKBONES = ("clip_vitb32", "siglip2_base")
_ABLATION_FUSIONS = ("token", "film", "cross_attn")
ABLATION_CONFIGS = {
    (backbone, fusion): make_config(fusion, backbone_type=backbone)
    for backbone in _ABLATION_BACKBONES for fusion in _ABLATION_FUSIONS
}
CONFIG_CLIP_FILM = ABLATION_CONFIGS[("clip_vitb32", "film")]
CONFIG_SIGLIP_FILM = ABLATION_CONFIGS[("siglip2_base", "film")]
CONFIG_CLIP_TOKEN = ABLATION_CONFIGS[("clip_vitb32", "token")]
CONFIG_CLIP_CROSS_ATTN = ABLATION_CONFIGS[("clip_vitb32", "cross_attn")]
CONFIG_SIGLIP_TOKEN = ABLATION_CONFIGS[("siglip2_base", "token")]
CONFIG_SIGLIP_CROSS_ATTN = ABLATION_CONFIGS[("siglip2_base", "cross_attn")]

# Import padrão do notebook: troque qual das linhas fica descomentada
# pra rodar cada experimento (experiment_name diferente -> checkpoints
# não se sobrescrevem).
CONFIG = CONFIG_TOKEN
# CONFIG = CONFIG_FILM
# CONFIG = CONFIG_CROSS_ATTN
# CONFIG = CONFIG_CLIP_FILM
# CONFIG = CONFIG_SIGLIP_FILM