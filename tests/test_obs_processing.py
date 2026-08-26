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


class TestSplitEpisodesStratified:
    def test_toda_tarefa_representada_em_val(self):
        from act_lang.data.libero import split_episodes_stratified

        # 5 tarefas, 10 episódios cada -- cenário Fase 2 (proporcionalmente)
        labels = {i: f"task_{i % 5}" for i in range(50)}
        episode_ids = list(range(50))

        train_ids, val_ids = split_episodes_stratified(episode_ids, labels, val_frac=0.2, seed=0)

        val_tasks = {labels[i] for i in val_ids}
        assert val_tasks == {f"task_{i}" for i in range(5)}, "toda tarefa deveria aparecer em val"
        assert set(train_ids) | set(val_ids) == set(episode_ids)
        assert set(train_ids) & set(val_ids) == set()

    def test_degenera_pra_split_simples_com_1_tarefa(self):
        """Com uma única tarefa (caso da Fase 1), o resultado tem que ser
        IDÊNTICO ao split_episodes plano -- sem essa garantia, usar o
        estratificado por padrão em ambas as fases seria arriscado."""
        from act_lang.data.libero import split_episodes, split_episodes_stratified

        episode_ids = list(range(23))
        labels = {i: "unica_tarefa" for i in episode_ids}

        tr_simples, va_simples = split_episodes(episode_ids, val_frac=0.15, seed=7)
        tr_estrat, va_estrat = split_episodes_stratified(episode_ids, labels, val_frac=0.15, seed=7)

        assert tr_simples == tr_estrat
        assert va_simples == va_estrat

    def test_tarefa_com_1_episodio_fica_em_train(self, capsys):
        from act_lang.data.libero import split_episodes_stratified

        labels = {0: "rara", 1: "comum", 2: "comum", 3: "comum", 4: "comum"}
        train_ids, val_ids = split_episodes_stratified(
            [0, 1, 2, 3, 4], labels, val_frac=0.3, seed=0
        )

        assert 0 in train_ids
        assert 0 not in val_ids
        captured = capsys.readouterr()
        assert "rara" in captured.out and "1 episódio" in captured.out


class TestSelectTaskIndexColumnCompat:
    """select_columns mudou de lugar entre versões do lerobot -- confirmado
    rodando local com lerobot 0.4.4 (AttributeError na versão anterior deste
    arquivo). O shim precisa acertar os dois formatos sem precisar saber
    qual versão está instalada."""

    def test_usa_select_columns_direto_quando_disponivel(self):
        from act_lang.data.libero import _select_task_index_column

        class DatasetVersaoAntiga:
            def select_columns(self, cols):
                return f"direto:{cols}"

        assert _select_task_index_column(DatasetVersaoAntiga()) == "direto:['task_index']"

    def test_cai_pra_hf_dataset_quando_select_columns_sumiu(self):
        from act_lang.data.libero import _select_task_index_column

        class HfDatasetInterno:
            def select_columns(self, cols):
                return f"via_hf_dataset:{cols}"

        class DatasetVersaoNova:
            hf_dataset = HfDatasetInterno()
            # sem select_columns direto -- igual ao lerobot 0.4.4

        assert (
            _select_task_index_column(DatasetVersaoNova())
            == "via_hf_dataset:['task_index']"
        )


class TestFitSemValidacao:
    """fit(val_loader=None): treino com 100% dos dados, sem early stopping,
    checkpoints periódicos -- avaliação de qual é o melhor fica pro rollout,
    fora deste módulo. Verifica: (a) o modo COM val continua idêntico ao de
    sempre; (b) o modo SEM val roda e salva o que deveria, sem tentar ler
    val_recon_z0 (que não existe nesse modo)."""

    def _setup(self, tmp_path):
        import torch
        from act_lang.models.act import ACT
        from act_lang.training.optim import build_optimizer

        torch.manual_seed(0)
        model = ACT(
            action_dim=7, state_dim=8, d_model=32, latent_dim=8, chunk_size=4,
            n_cameras=2, n_encoder_layers=1, n_decoder_layers=1, n_heads=2,
            pretrained_backbone=False,
        )
        optimizer = build_optimizer(model, lr=1e-3, lr_backbone=1e-3)

        def fake_bridge(batch, device):
            b = 2
            images = torch.rand(b, 2, 3, 16, 16, device=device)
            state = torch.rand(b, 8, device=device)
            actions = torch.rand(b, 4, 7, device=device)
            is_pad = torch.zeros(b, 4, dtype=torch.bool, device=device)
            return images, state, actions, is_pad, None

        loader = [{"dummy": 1}, {"dummy": 2}]  # bridge ignora o conteúdo
        return model, optimizer, fake_bridge, loader, tmp_path / "ckpt"

    def test_com_val_loader_comportamento_identico_ao_original(self, tmp_path):
        import torch
        from act_lang.training.loop import fit

        model, optimizer, bridge, loader, ckpt_dir = self._setup(tmp_path)
        device = torch.device("cpu")

        history = fit(
            model, loader, loader, bridge, optimizer, device,
            checkpoint_dir=ckpt_dir, num_epochs=2, patience=40, checkpoint_every=100,
        )

        assert set(history.keys()) == {
            "train_loss", "train_recon", "train_kld", "train_mu_abs_mean",
            "val_loss", "val_recon", "val_kld", "val_recon_z0", "val_mu_abs_mean",
        }
        assert len(history["val_recon_z0"]) == 2
        # mu_abs_mean é uma média de valor absoluto -- nunca negativo, sempre finito
        assert all(v >= 0 and v == v for v in history["train_mu_abs_mean"])  # v==v descarta NaN
        assert all(v >= 0 and v == v for v in history["val_mu_abs_mean"])
        # seleção por val ainda salva ao menos um best_epoch*.pt
        assert any(ckpt_dir.glob("best_epoch*.pt"))

    def test_sem_val_loader_roda_e_nao_quebra(self, tmp_path):
        import torch
        from act_lang.training.loop import fit

        model, optimizer, bridge, loader, ckpt_dir = self._setup(tmp_path)
        device = torch.device("cpu")

        history = fit(
            model, loader, None, bridge, optimizer, device,
            checkpoint_dir=ckpt_dir, num_epochs=3, checkpoint_every=1,
        )

        assert set(history.keys()) == {
            "train_loss", "train_recon", "train_kld", "train_mu_abs_mean",
        }
        assert len(history["train_loss"]) == 3
        # sem val: nenhum best_epoch*.pt (não há critério de seleção offline)
        assert not any(ckpt_dir.glob("best_epoch*.pt"))
        # mas os periódicos (checkpoint_every=1) e o last_checkpoint continuam
        assert (ckpt_dir / "last_checkpoint.pt").exists()
        assert len(list(ckpt_dir.glob("periodic_epoch*.pt"))) == 3

    def test_resume_com_history_de_esquema_antigo_nao_quebra(self, tmp_path):
        """Simula retomar de um checkpoint salvo por uma versão anterior
        deste arquivo, cujo history não tinha 'mu_abs_mean'. O setdefault()
        em fit() precisa lidar com isso sem KeyError."""
        import torch
        from act_lang.training.loop import fit

        model, optimizer, bridge, loader, ckpt_dir = self._setup(tmp_path)
        device = torch.device("cpu")

        history_antigo = {
            "train_loss": [0.5], "train_recon": [0.4], "train_kld": [0.1],
            "val_loss": [0.6], "val_recon": [0.5], "val_kld": [0.1], "val_recon_z0": [0.5],
            # sem train_mu_abs_mean / val_mu_abs_mean -- esquema de antes desta mudança
        }

        history = fit(
            model, loader, loader, bridge, optimizer, device,
            checkpoint_dir=ckpt_dir, num_epochs=2, start_epoch=1,
            history=history_antigo,
        )

        # a época 0 (antiga) não tem mu_abs_mean; só a nova (época 1) tem
        assert len(history["train_loss"]) == 2  # antiga + nova
        assert len(history["train_mu_abs_mean"]) == 1  # só a época nova


class TestSplitEpisodesMinHoldout:
    def test_reserva_exatamente_n_por_tarefa(self):
        from act_lang.data.libero import split_episodes_min_holdout

        # 10 tarefas, 5 episódios cada -- proporção real da Fase 2
        labels = {i: f"task_{i % 10}" for i in range(50)}
        episode_ids = list(range(50))

        train_ids, val_ids = split_episodes_min_holdout(
            episode_ids, labels, n_val_per_task=1, seed=0
        )

        assert len(val_ids) == 10  # exatamente 1 por tarefa
        val_tasks = sorted(labels[i] for i in val_ids)
        assert val_tasks == sorted(f"task_{i}" for i in range(10))  # todas representadas
        assert set(train_ids) | set(val_ids) == set(episode_ids)
        assert set(train_ids) & set(val_ids) == set()
        assert len(train_ids) == 40  # 4 por tarefa sobram pro treino

    def test_tarefa_com_1_episodio_fica_toda_em_train(self, capsys):
        from act_lang.data.libero import split_episodes_min_holdout

        labels = {0: "rara", 1: "comum", 2: "comum", 3: "comum"}
        train_ids, val_ids = split_episodes_min_holdout(
            [0, 1, 2, 3], labels, n_val_per_task=1, seed=0
        )

        assert 0 in train_ids and 0 not in val_ids
        assert len(val_ids) == 1  # só "comum" contribui
        captured = capsys.readouterr()
        assert "rara" in captured.out and "1 episódio" in captured.out

    def test_nunca_deixa_uma_tarefa_sem_dado_de_treino(self):
        """Mesmo pedindo n_val_per_task=1 numa tarefa com EXATAMENTE 2
        episódios, pelo menos 1 tem que sobrar pro treino."""
        from act_lang.data.libero import split_episodes_min_holdout

        labels = {0: "so_2", 1: "so_2"}
        train_ids, val_ids = split_episodes_min_holdout(
            [0, 1], labels, n_val_per_task=1, seed=0
        )
        assert len(train_ids) == 1
        assert len(val_ids) == 1
