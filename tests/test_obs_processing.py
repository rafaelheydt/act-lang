"""Testes das conversões críticas do rollout.

Os casos do quat2axisangle são os que foram validados manualmente contra a
fórmula do LiberoProcessorStep — agora automatizados: se a convenção mudar,
`pytest` acusa antes do rollout falhar silenciosamente.
"""

from collections import deque

import numpy as np

from act_lang.eval.obs_processing import quat2axisangle
from act_lang.eval.rollout_libero import temporal_ensemble


class TestQuat2AxisAngle:
    def test_identidade(self):
        # quat identidade (x,y,z,w) = (0,0,0,1) -> rotação nula
        np.testing.assert_allclose(
            quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3), atol=1e-8
        )

    def test_90_graus_em_z(self):
        # 90° em torno de z: (0, 0, sin(45°), cos(45°)) -> (0, 0, pi/2)
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        np.testing.assert_allclose(
            quat2axisangle(np.array([0.0, 0.0, s, c])),
            np.array([0.0, 0.0, np.pi / 2]),
            atol=1e-7,
        )

    def test_180_graus_em_x(self):
        # 180° em torno de x: (1, 0, 0, 0) -> (pi, 0, 0)
        np.testing.assert_allclose(
            quat2axisangle(np.array([1.0, 0.0, 0.0, 0.0])),
            np.array([np.pi, 0.0, 0.0]),
            atol=1e-7,
        )

    def test_w_fora_do_intervalo_nao_explode(self):
        # erros numéricos podem produzir |w| ligeiramente > 1 — o clip protege
        out = quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0000001]))
        np.testing.assert_allclose(out, np.zeros(3), atol=1e-6)


class TestTemporalEnsemble:
    def test_indices_e_direcao_dos_pesos(self):
        """Chunk predito há k passos contribui com seu índice k; w0 = mais antigo."""
        chunk_size, action_dim = 4, 2
        buffer = deque(maxlen=chunk_size)
        # chunk_t[i] = valor único que identifica (chunk, índice)
        for t in range(3):  # 3 chunks: t=0 (mais antigo), 1, 2 (mais novo)
            chunk = np.arange(chunk_size * action_dim, dtype=float).reshape(
                chunk_size, action_dim
            ) + 100 * t
            buffer.append(chunk)

        m = 0.5
        result = temporal_ensemble(buffer, m)

        # esperado: buffer[0] (antigo) -> índice 2; buffer[1] -> 1; buffer[2] (novo) -> 0
        preds_esperados = np.stack([buffer[0][2], buffer[1][1], buffer[2][0]])
        w = np.exp(-m * np.arange(3))
        w /= w.sum()
        np.testing.assert_allclose(result, w @ preds_esperados)
        assert w[0] > w[-1], "peso maior deve ser da predição mais antiga (paper)"

    def test_buffer_unitario(self):
        buffer = deque(maxlen=4)
        chunk = np.ones((4, 2))
        buffer.append(chunk)
        np.testing.assert_allclose(temporal_ensemble(buffer, m=0.01), chunk[0])


class TestDecoderStyleEquivalence:
    """decoder_style='detr' é uma ablação opt-in -- o default ('torch') não
    pode mudar de comportamento por causa dela. Trava isso automaticamente."""

    def _make_model(self, decoder_style="torch"):
        from act_lang.models.act import ACT
        return ACT(
            action_dim=7, state_dim=8, d_model=64, latent_dim=16, chunk_size=6,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=4,
            pretrained_backbone=False, decoder_style=decoder_style,
        )

    def test_default_identico_a_torch_explicito(self):
        import torch

        torch.manual_seed(0)
        model_default = self._make_model()  # sem passar decoder_style
        torch.manual_seed(0)
        model_explicit = self._make_model(decoder_style="torch")

        model_default.eval()
        model_explicit.eval()
        images = torch.rand(2, 2, 3, 64, 64)
        state = torch.rand(2, 8)
        actions = torch.rand(2, 6, 7)
        is_pad = torch.zeros(2, 6, dtype=torch.bool)

        with torch.no_grad():
            p1, _, _ = model_default(
                images, state, actions=actions, is_pad=is_pad, sample_posterior=False
            )
            p2, _, _ = model_explicit(
                images, state, actions=actions, is_pad=is_pad, sample_posterior=False
            )
        import numpy as np
        np.testing.assert_array_equal(p1.numpy(), p2.numpy())

    def test_detr_style_roda_e_tem_gradiente(self):
        import torch

        model = self._make_model(decoder_style="detr")
        images = torch.rand(2, 2, 3, 64, 64)
        state = torch.rand(2, 8)
        actions = torch.rand(2, 6, 7)
        is_pad = torch.zeros(2, 6, dtype=torch.bool)

        pred, mu, logvar = model(images, state, actions=actions, is_pad=is_pad)
        assert pred.shape == (2, 6, 7)
        (pred.sum() + mu.sum() + logvar.sum()).backward()
        assert model.action_queries.grad is not None
        assert model.action_queries.grad.abs().sum() > 0


class TestRolloutInitStateSelection:
    """Sem MuJoCo real: um env falso registrando só as chamadas de reset()
    e checando quais init_state_id foram usados, na ordem certa."""

    def _run_com_env_falso(self, **kwargs):
        import torch
        from act_lang.data.normalize import MinMaxNormalizer
        from act_lang.eval.rollout_libero import rollout_libero

        class ModeloFalso:
            chunk_size = 4

            def eval(self):
                pass

            def __call__(self, images, state, actions=None, task_texts=None):
                return torch.zeros(1, self.chunk_size, 7), None, None

        class EnvFalso:
            def __init__(self, n_states=50):
                self._init_states = list(range(n_states))
                self.init_state_id = 0
                self.chamadas = []

            def reset(self, seed=None):
                self.chamadas.append(self.init_state_id)
                img = np.zeros((8, 8, 3), dtype=np.uint8)
                return {"pixels": {"image": img, "image2": img}}, {"is_success": True}

            def step(self, action):
                img = np.zeros((8, 8, 3), dtype=np.uint8)
                obs = {"pixels": {"image": img, "image2": img}}
                return obs, 0.0, True, False, {"is_success": True}

        # process_libero_obs espera robot_state -- monkeypatch pra pular a
        # conversão real (não é o que este teste está verificando).
        import act_lang.eval.rollout_libero as rl_module
        original = rl_module.process_libero_obs
        rl_module.process_libero_obs = lambda raw_obs, sn, dev: (
            torch.zeros(1, 2, 3, 8, 8), torch.zeros(1, 8)
        )
        try:
            env = EnvFalso()
            norm = MinMaxNormalizer(x_min=torch.zeros(7), x_range=torch.ones(7))
            rollout_libero(
                ModeloFalso(), env, norm, norm, torch.device("cpu"),
                video_dir="/tmp/test_rollout_videos", **kwargs,
            )
            return env.chamadas
        finally:
            rl_module.process_libero_obs = original

    def test_comportamento_antigo_preservado(self):
        """Sem init_state_ids: continua sendo init_state_start + range(n_episodes)."""
        chamadas = self._run_com_env_falso(n_episodes=5, init_state_start=10)
        assert chamadas == [10, 11, 12, 13, 14]

    def test_lista_explicita_espalhada(self):
        """Com init_state_ids: usa exatamente essa lista, na ordem dada."""
        chamadas = self._run_com_env_falso(init_state_ids=[0, 15, 30, 45])
        assert chamadas == [0, 15, 30, 45]


class TestPickDevice:
    def test_preferencia_explicita_sempre_vence(self, monkeypatch):
        import torch
        from act_lang.utils.runtime import pick_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        # mesmo se a GPU 0 tivesse mais memória livre, index explícito=1 vence
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (999_000_000_000, 1_000_000_000_000))
        assert pick_device(preferred_index=1) == torch.device("cuda:1")

    def test_escolhe_gpu_com_mais_memoria_livre(self, monkeypatch):
        import torch
        from act_lang.utils.runtime import pick_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

        def fake_mem_get_info(i):
            # simula o seu nvidia-smi: cuda:0 (A2000, 6GB, quase cheia de
            # outros processos) tem menos livre que cuda:1 (RTX 3050, 8GB)
            return {0: (500_000_000, 6_000_000_000), 1: (7_500_000_000, 8_000_000_000)}[i]

        monkeypatch.setattr(torch.cuda, "mem_get_info", fake_mem_get_info)
        assert pick_device() == torch.device("cuda:1")

    def test_sem_cuda_cai_pra_cpu(self, monkeypatch):
        import torch
        from act_lang.utils.runtime import pick_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert pick_device() == torch.device("cpu")
