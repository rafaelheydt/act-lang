# act-lang

ACT (Action Chunking Transformer) com condicionamento em linguagem — pesquisa de
mestrado comparando mecanismos de fusão (token simples / FiLM / cross-attention),
com validação no Push-T e benchmark principal no LIBERO.

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
!git clone https://github.com/SEU_USER/act-lang.git /content/act-lang
%cd /content/act-lang
!pip install -q -e . "lerobot[libero]"

%load_ext autoreload
%autoreload 2
```

Com `%autoreload 2`, edite os `.py` (editor do Colab ou local + `!git pull`) e a
mudança vale na célula seguinte, sem reiniciar o runtime. Checkpoints e vídeos
vão para o Drive (estão no `.gitignore`).

Notebooks finos em `notebooks/`: `01_treino_libero.ipynb` (dados + treino) e
`02_rollout_libero.ipynb` (avaliação no ambiente).

## Testes

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Cobrem as duas conversões críticas do rollout (fórmula `quat2axisangle` do
`LiberoProcessorStep` e a direção dos pesos do temporal ensembling).

## Correções aplicadas em relação ao notebook original (v1)

Registradas nos docstrings dos módulos; resumo:

| # | Correção | Onde | Exige retreino? |
|---|----------|------|-----------------|
| 1 | Máscara de padding no CVAE encoder (`src_key_padding_mask`) | `models/act.py` | sim |
| 2 | Normalização ImageNet como buffer do backbone | `models/backbone.py` | sim |
| 3 | Temporal ensembling: w0 = predição mais antiga (paper) | `eval/rollout_libero.py` | não |
| 4 | Validação determinística (z=mu) + seleção por `val_recon_z0` | `training/loop.py` | não |
| 5 | LR do backbone 10x menor + FrozenBatchNorm opcional | `training/optim.py`, `models/backbone.py` | sim |
| 6 | Pos. embeddings para tokens de state e z | `models/act.py` | sim |
| 7 | Off-by-one no resume de checkpoint | `training/checkpoints.py` | — |
| 8 | Epsilon no range da normalização min-max | `data/normalize.py` | — |

**Atenção**: pelas correções 1, 2, 5 e 6, checkpoints do notebook v1 são
incompatíveis com este código (state_dict tem chaves novas). O baseline deve
ser retreinado — e os números v1/v2 não são comparáveis entre si.
