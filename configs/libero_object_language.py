"""Config: LIBERO, 10 tarefas do libero_object, COM linguagem (Fase 3).

Base idêntica à Fase 2 (mesmas 10 tarefas, mesmo split min_holdout) -- só
adiciona a fusão de linguagem. Comparar contra o "chão" da Fase 2
(experiment_name="libero_v2_10tasks_sem_lingua") é o que dá sentido
científico a esta fase: quanto a linguagem melhora em relação a não ter
instrução nenhuma? E comparar os 3 experimentos abaixo entre si é a
pergunta central da dissertação: qual mecanismo de fusão funciona melhor?

`make_config(fusion_type)` gera os 3 configs a partir de uma base
compartilhada -- evita triplicar ~60 linhas quase idênticas, mas cada
CONFIG_* continua sendo um dict comum, do jeito que o resto do projeto
(notebooks, fit(), etc.) já espera.
"""

from configs.libero_object_multitask import TASK_TEXTS_LIBERO_OBJECT_10


def make_config(fusion_type: str) -> dict:
    assert fusion_type in ("token", "film", "cross_attn"), fusion_type
    return {
        "experiment_name": f"libero_v2_10tasks_lingua_{fusion_type}",
        "device_index": None,  # None = auto (GPU com mais memória livre)
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
        "free_bits": 0.0,  # 0.0 = fiel ao KL cru do ACT oficial (sem free bits)
        "grad_clip_norm": 10.0,
        "checkpoint_every": 50,
        # rollout -- mesma ressalva da Fase 2: com 10 tarefas, cada uma
        # precisa de env/task_id próprio (notebook de rollout ainda não
        # generalizado pra multi-task).
        "rollout_m": 0.01,
        "rollout_max_steps": 300,
        "rollout_n_episodes": 10,
    }


CONFIG_TOKEN = make_config("token")
CONFIG_FILM = make_config("film")
CONFIG_CROSS_ATTN = make_config("cross_attn")

# Import padrão do notebook: troque qual das 3 linhas fica descomentada
# pra rodar cada experimento (experiment_name diferente -> checkpoints
# não se sobrescrevem).
CONFIG = CONFIG_TOKEN
# CONFIG = CONFIG_FILM
# CONFIG = CONFIG_CROSS_ATTN