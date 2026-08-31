"""Config: baseline LIBERO, 1 tarefa, sem linguagem (Fase 1).

Dict Python simples: sem dependência extra, importável no notebook e fácil de
copiar/variar por experimento (um arquivo por experimento em configs/).
"""

CONFIG = {
    "experiment_name": "libero_v2_tarefa_unica_sem_lingua",
    "device_index": None,  # None = auto (GPU com mais memória livre).
    # Fixe um índice (ex: 1) pra forçar uma GPU específica -- útil numa
    # máquina com várias GPUs de tamanhos diferentes.
    # dados
    "task_texts": {"pick up the milk and place it in the basket"},
    "task_suite_name": "libero_object",
    "val_frac": 0.1,
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
    "freeze_bn": True,  # FrozenBatchNorm2d no backbone (padrão ACT/DETR)
    "decoder_style": "torch",  # "torch" = baseline (validado); "detr" = ablação
    # com reinjeção posicional por camada -- rodar como experimento separado,
    # nunca junto com uma mudança de mais nada (ver models/decoder_detr.py)
    # otimização
    "lr": 1e-4,
    "lr_backbone": 1e-5,  # 10x menor — padrão ACT original
    "weight_decay": 1e-4,
    "num_epochs": 300,
    "kl_weight": 10.0,
    "free_bits": 0.0,  # 0.0 = fiel ao KL cru do ACT oficial (sem free bits)
    "grad_clip_norm": 10.0,
    "checkpoint_every": 50,
    # rollout
    "rollout_m": 0.01,
    "rollout_max_steps": 300,
    "rollout_n_episodes": 10,
}