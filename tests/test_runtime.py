"""Testes de act_lang.utils.runtime.

Foco: get_checkpoint_dir não pode chamar drive.mount() quando o Drive já
está montado no disco -- drive.mount() exige o kernel IPython vivo (fala
com o frontend via mensagem) e falha com AttributeError quando chamado de
um subprocesso (ex.: notebook que roda scripts/train.py via
subprocess.Popen em vez de reimplementar o treino inline -- ver
docstring de get_checkpoint_dir). Como o mountpoint é real no nível do
SO, um subprocesso deve conseguir usá-lo sem precisar (re)montar.
"""

import sys
import types

import pytest


@pytest.fixture()
def fake_colab_drive(monkeypatch, tmp_path):
    """Injeta um `google.colab.drive` falso cujo mount() explode se
    chamado -- simula estar rodando "no Colab" (is_colab()==True) sem
    depender do pacote real nem de um kernel IPython de verdade."""
    mydrive = tmp_path / "drive" / "MyDrive"

    calls = []
    fake_drive_module = types.ModuleType("google.colab.drive")

    def fake_mount(mountpoint, force_remount=False):
        calls.append((mountpoint, force_remount))
        raise AttributeError(
            "'NoneType' object has no attribute 'kernel'"
        )  # reproduz o erro real de subprocesso sem kernel

    fake_drive_module.mount = fake_mount

    fake_google = types.ModuleType("google")
    fake_colab = types.ModuleType("google.colab")
    fake_google.colab = fake_colab
    fake_colab.drive = fake_drive_module

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setitem(sys.modules, "google.colab.drive", fake_drive_module)

    # get_checkpoint_dir usa o path fixo "/content/drive/MyDrive" -- para
    # testar sem tocar o filesystem real, redirecionamos via monkeypatch
    # do próprio Path usado dentro da função (ver teste abaixo).
    return mydrive, calls


class TestGetCheckpointDir:
    def test_nao_monta_se_ja_montado(self, monkeypatch, fake_colab_drive):
        """Mountpoint já existe -> drive.mount() NUNCA é chamado."""
        mydrive, calls = fake_colab_drive
        mydrive.mkdir(parents=True)  # simula Drive já montado pela célula

        from act_lang.utils import runtime
        monkeypatch.setattr(runtime, "is_colab", lambda: True)
        monkeypatch.setattr(
            runtime, "Path",
            lambda p: mydrive if str(p) == "/content/drive/MyDrive" else __import__("pathlib").Path(p),
        )

        result = runtime.get_checkpoint_dir("meu_experimento")

        assert calls == [], "drive.mount() foi chamado mesmo com o Drive já montado"
        assert result == mydrive / "meu_experimento"
        assert result.is_dir()

    def test_monta_se_ainda_nao_montado(self, monkeypatch, fake_colab_drive):
        """Mountpoint NÃO existe -> tenta montar (comportamento original,
        preservado para quando a célula do notebook ainda não montou)."""
        mydrive, calls = fake_colab_drive
        # NÃO cria mydrive -- simula Drive ainda não montado

        from act_lang.utils import runtime
        monkeypatch.setattr(runtime, "is_colab", lambda: True)
        monkeypatch.setattr(
            runtime, "Path",
            lambda p: mydrive if str(p) == "/content/drive/MyDrive" else __import__("pathlib").Path(p),
        )

        with pytest.raises(AttributeError, match="kernel"):
            runtime.get_checkpoint_dir("meu_experimento")

        assert len(calls) == 1, "drive.mount() deveria ter sido tentado"

    def test_fora_do_colab_ignora_drive_por_completo(self, monkeypatch, tmp_path, fake_colab_drive):
        """is_colab()==False -> nem olha para /content/drive; usa local_base."""
        _, calls = fake_colab_drive
        from act_lang.utils import runtime
        monkeypatch.setattr(runtime, "is_colab", lambda: False)

        result = runtime.get_checkpoint_dir("meu_experimento", local_base=tmp_path)

        assert calls == []
        assert result == tmp_path / "meu_experimento"
        assert result.is_dir()
