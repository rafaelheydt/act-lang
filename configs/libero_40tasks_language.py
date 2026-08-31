"""Config: LIBERO, 40 tarefas (todo o repositório lerobot/libero disponível),
COM linguagem.

Mesma estrutura de configs/libero_object_language.py, mas com o conjunto de
tarefas ampliado: as 10 tarefas do libero_object originais + ~30 tarefas
adicionais já presentes no mesmo repositório HuggingFace (lerobot/libero) --
gavetas, fogão, moka pot, micro-ondas, etc; não é uma suite "pura" do LIBERO
oficial, é o que esse repositório específico já hospeda. 1.693 episódios no
total (vs. 454 na versão de 10 tarefas), ~4x mais dado -- objetivo direto:
reduzir o overfitting que apareceu cedo na versão de 10 tarefas (ver
diagnóstico de 30/08, val_recon divergindo de train_recon já na época 3).

`make_config(fusion_type)` gera os 3 configs a partir de uma base
compartilhada, mesmo padrão de libero_object_language.py.
"""

TASK_TEXTS_LIBERO_40 = {
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
    "pick up the book and place it in the back compartment of the caddy",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "push the plate to the front of the stove",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "put the yellow and white mug in the microwave and close it",
    "turn on the stove",
    "turn on the stove and put the moka pot on it",
}


def make_config(fusion_type: str) -> dict:
    assert fusion_type in ("token", "film", "cross_attn"), fusion_type
    return {
        "experiment_name": f"libero_v2_40tasks_lingua_{fusion_type}",
        "device_index": None,  # None = auto (GPU com mais memória livre)
        # dados
        "task_texts": TASK_TEXTS_LIBERO_40,
        "task_suite_name": "libero_40_mixed",  # não é suite oficial pura -- ver docstring
        "val_frac": 0.1,  # usado só se val_strategy="fraction"
        "val_strategy": "min_holdout",
        "n_val_per_task": 1,  # 40 tarefas -> 40 episódios de val (vs. 10 antes)
        "seed": 42,
        "obs_horizon": 1,
        "pred_horizon": 50,
        "batch_size": 32,
        # modelo -- idêntico à versão de 10 tarefas (mesma arquitetura,
        # só muda a quantidade de dado)
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
        # otimização
        "lr": 1e-4,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "num_epochs": 50,  # validação mais simples/rápida antes de comprometer com 300
        "kl_weight": 10.0,
        "kl_warmup_epochs": 10,  # annealing linear 0->kl_weight -- ver diagnóstico de 31/08 (decoder ignorando z)
        "free_bits": 0.05,  # reativado -- ver diagnóstico de colapso de posterior em 30/08
        "grad_clip_norm": 10.0,
        "checkpoint_every": 50,
        # rollout -- com 40 tarefas, ainda mais que a versão de 10: cada uma
        # precisa de env/task_id próprio (mesma ressalva das configs anteriores).
        "rollout_m": 0.01,
        "rollout_max_steps": 300,
        "rollout_n_episodes": 10,
    }


CONFIG_TOKEN = make_config("token")
CONFIG_FILM = make_config("film")
CONFIG_CROSS_ATTN = make_config("cross_attn")

CONFIG = CONFIG_FILM