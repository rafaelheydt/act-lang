# act-lang

ACT (Action Chunking Transformer) com condicionamento em linguagem — pesquisa de
mestrado comparando mecanismos de fusão (token simples / FiLM / cross-attention),
com validação no benchmark principal no LIBERO.

Para como o código funciona por dentro (arquitetura bloco a bloco, loop de
treinamento), veja [ARCHITECTURE.md](ARCHITECTURE.md).

## Fases

1. **Baseline LIBERO, 1 tarefa, sem linguagem** (atual) — valida arquitetura e
   pipeline sem ambiguidade. Config: `configs/libero_single_task.py`.
2. **10 tarefas misturadas, sem linguagem** — mede o "chão" (quanto a ambiguidade
   custa sem instrução).
3. **Fusão de linguagem** — mecanismos em `src/act_lang/models/fusion/`, todos
   implementando a interface de `fusion/base.py`; trocar mecanismo = 1 linha no
   config.

## Uso no Colab

```python
%cd /content                # sai da pasta antes de apagá-la (evita quebrar o cwd do shell)
!rm -rf /content/act-lang   # evita clone silenciosamente ignorado numa pasta antiga
!git clone https://github.com/rafaelheydt/act-lang.git /content/act-lang
%cd /content/act-lang
!pip install -q -e . "lerobot[libero]"

import sys
if "/content/act-lang" not in sys.path:
    sys.path.insert(0, "/content/act-lang")  # necessário p/ importar configs/ (fora de src/)
```

**Depois que essa célula rodar pela primeira vez em cada runtime novo**, faça
`Runtime > Restart session` antes de continuar — o Python só lê o registro do
`pip install -e .` na inicialização do interpretador, não em tempo real. Em
seguida rode a célula de novo (rápido, o cache do pip já tem tudo) e siga
normalmente. Sem esse restart, `import act_lang` falha mesmo com tudo
instalado certo.

Sem `%autoreload`: o IPython pinado pelo pacote `google-colab` (7.34.0) quebra
com ele no runtime atual, e forçar upgrade do IPython quebra `drive.mount()` e
exibição de vídeo em troca. Editou algo em `src/act_lang/`? Restart session +
rodar a célula de setup de novo. Checkpoints e vídeos vão para o Drive (estão
no `.gitignore`).

Notebooks finos em `notebooks/`: `01_treino_libero.ipynb` (dados + treino) e
`02_rollout_libero.ipynb` (avaliação no ambiente).

## Uso local (ex: quando a sessão do Colab expira)

Os dois notebooks detectam o ambiente automaticamente (`try: import
google.colab`) e ajustam sozinhos: sem clone/reinstalação a cada vez,
checkpoints numa pasta local em vez do Drive, e a GPU escolhida
automaticamente entre as disponíveis (a de mais memória livre no momento,
salvo se você fixar `device_index` no config).

Pré-requisito, uma vez, no terminal (dentro do seu ambiente conda):

```bash
git clone https://github.com/rafaelheydt/act-lang.git
cd act-lang && pip install -e . "lerobot[libero]"
```

Depois é só abrir o Jupyter/notebook normalmente e rodar as células — sem
precisar do restart de kernel que o Colab exige (`pip install -e .` já
funciona no mesmo processo fora daquele ambiente específico).

Checkpoints vão para `~/act-lang-checkpoints/<experiment_name>/` por padrão;
mude com a variável de ambiente `ACT_LANG_CHECKPOINT_DIR` se preferir outro
lugar. Pra forçar uma GPU específica (ex: a RTX 3050 no índice 1, numa
máquina com mais de uma GPU), edite `configs/libero_single_task.py`:

```python
"device_index": 1,  # None = automático (mais memória livre)
```

## Testes

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

