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
