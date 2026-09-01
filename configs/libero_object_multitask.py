"""Config: LIBERO, 10 tarefas misturadas do libero_object, sem linguagem (Fase 2).

Mede o "chão": quanto a ambiguidade custa quando o modelo vê a MESMA cena
(mesmos objetos na mesa) mas não recebe instrução nenhuma dizendo qual pegar.
Sem essa informação, não há como saber -- diferente da Fase 1 (1 tarefa),
onde a ambiguidade era zero por construção.

As 10 tarefas confirmadas rodando ao vivo (benchmark.get_benchmark_dict()
["libero_object"]().get_task(i).language, i=0..9) -- não adivinhadas:
"""

TASK_TEXTS_LIBERO_OBJECT_10 = {
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",  # a mesma da Fase 1
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
}

CONFIG = {
    "experiment_name": "libero_v2_10tasks_sem_lingua",
    "device_index": None,  # None = auto (GPU com mais memória livre).
    # dados
    "task_texts": TASK_TEXTS_LIBERO_OBJECT_10,
    "task_suite_name": "libero_object",
    "val_frac": 0.1,  # usado só se val_strategy="fraction"
    "val_strategy": "min_holdout",  # "fraction" (Fase 1) | "min_holdout" (aqui)
    "n_val_per_task": 1,  # reserva 1 episódio "nunca visto" por tarefa pra
    # val -- com ~5 episódios/tarefa, isso deixa a métrica de validação
    # comparável entre as 10 tarefas, maximizando o que sobra pro treino.
    "seed": 42,
    "obs_horizon": 1,
    "pred_horizon": 50,
    "batch_size": 32,
    # modelo -- MESMA arquitetura da Fase 1 (sem fusão de linguagem: fusion=None
    # no ACT continua o baseline; task_texts do batch são ignorados pelo modelo,
    # só usados aqui pro filtro/split de dados)
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
    "decoder_style": "torch",
    # otimização
    "lr": 1e-4,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "num_epochs": 300,
    "kl_weight": 10.0,
    "free_bits": 0.05,
    "grad_clip_norm": 10.0,
    "checkpoint_every": 50,
    # rollout -- COM 10 tarefas, cada uma precisa de env/task_id próprio;
    # ver notebooks/02_rollout_libero.ipynb, célula precisa iterar por task
    # (ainda não generalizado pra multi-task -- fazer antes do rollout da Fase 2)
    "rollout_m": 0.01,
    "rollout_max_steps": 300,
    "rollout_n_episodes": 10,
}
