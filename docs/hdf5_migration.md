
## Migração JPEG → HDF5 nativo

Contexto completo do porquê da migração (benchmarks de I/O, riscos, plano de
implementação original) está fora deste arquivo -- isto documenta os passos
de verificação feitos antes/durante a implementação: o mapeamento das 40
tarefas (risco #3 do plano), a checagem de convenção de ação (risco #2, o
mais importante), um achado adicional sobre orientação de imagem (seção 3),
e a implementação em si (seção 4).

**Resolução: 128×128 nativo** (decisão de risco #1 do plano, tomada
explicitamente com o usuário em 2026-09-03) -- sem upsampling pra 256×256.
Ganho de I/O completo; muda o experimento (menos detalhe visual/tokens de
imagem que o pipeline JPEG atual), então comparação de curva de loss entre
os dois pipelines não é 1:1 -- ver treino piloto (passo 7, ainda não feito).

### 1. Mapeamento das 40 tarefas → arquivos HDF5 oficiais

Resultado completo em [`configs/task_mapping_libero40.json`](../configs/task_mapping_libero40.json).

As 40 `task_texts` de `configs/libero_40tasks_language.py` batem, quase todas
por igualdade exata de string (normalizando o nome de arquivo -- minúsculas,
`_` → espaço, removendo o prefixo `{CENA}_SCENE{n}_`), com exatamente:

| Suíte oficial | Tarefas usadas |
|---|---|
| `libero_object` | 10 / 10 (completa) |
| `libero_spatial` | 10 / 10 (completa) |
| `libero_goal` | 10 / 10 (completa) |
| `libero_10` | 10 / 10 (completa) |
| `libero_90` | 0 / 89 |

Ou seja: as 40 tarefas do repositório `lerobot/libero` são simplesmente a
união de 4 das 5 suítes oficiais do LIBERO, completas -- não é um
subconjunto arbitrário. `libero_90` não contribui nenhuma tarefa.

Duas tarefas tiveram texto idêntico a arquivos de mais de uma suíte (mesma
frase, cena diferente -- coincidência entre o texto de uma tarefa de
`libero_10`/`libero_goal` e uma tarefa de `libero_90` com layout diferente):

- **"pick up the book and place it in the back compartment of the caddy"**:
  existe em `libero_10/STUDY_SCENE1` e `libero_90/STUDY_SCENE2`. Contagem de
  demos não desambigua (50 em ambos). Resolvido pelo padrão acima: as outras
  9 tarefas de `libero_10` já batiam exato, e `libero_90` não contribui
  nenhuma outra tarefa no conjunto de 40 -- forte evidência de que é
  `libero_10`.
- **"turn on the stove"**: existe em `libero_goal`, `libero_90/KITCHEN_SCENE3`
  e `libero_90/KITCHEN_SCENE9`. Mesmo raciocínio: as outras 9 tarefas de
  `libero_goal` já batiam exato → é `libero_goal`.

Ambas as resoluções ficam marcadas com `"match": "resolved_by_clustering"` e
uma nota no JSON, para rastreabilidade -- não foram assumidas, foram
inferidas da estrutura do próprio conjunto de 40 e documentadas como tal.

### 2. Convenção de ação -- verificação estatística (risco mais importante do plano)

**Resultado: HDF5 oficial e o dataset `lerobot/libero` usam a MESMA
convenção de ação.** Não há conversão de espaço de ação necessária.

Evidência:

- `lerobot/libero` (`meta/stats.json`, stats globais das 273465 frames,
  key `"action"`, shape 7): dims 0-2 com `min≈-0.9375`/`max≈0.9375`
  (clipping simétrico), dims 3-5 com faixa pequena (~±0.3-0.4), dim 6
  (gripper) com `min=-1.0`/`max=1.0` exatos.
- Arquivo HDF5 oficial lido remotamente (`libero_object/pick_up_the_alphabet_soup_..._demo.hdf5`,
  demo 0, via leitura HTTP com range-request usando `fsspec`+`h5py`, sem
  baixar o arquivo inteiro): `actions` shape `(148, 7)` float64, com exatamente
  o mesmo padrão -- dims 0-2 limitadas a `[-0.9375, 0.9375]`, dims 3-5
  pequenas, dim 6 estritamente `{-1.0, 1.0}` (gripper binário, não contínuo).

Isso bate com a documentação do controlador `OSC_POSE` do `robosuite`
(confirmado via `env_wrapper.py` do repositório oficial do LIBERO, que
instancia `"OSC_POSE"` como `default_controller`): 6 DOF fixos (delta
posição xyz + delta orientação em eixo-ângulo, ângulo em radianos, eixos no
frame global) + 1 dimensão de gripper. `lerobot/libero` claramente preservou
a ação crua do robosuite sem reconversão -- **não** é a convenção de EE
absoluto + quaternion que o risco #2 do plano cogitava como possibilidade
(baseado no padrão de outras conversões da família `lerobot/libero`-like,
como `GT-111/libero_v3_eef`). Essa hipótese foi descartada por evidência
direta, não por suposição.

Também confirmado nesta verificação: `obs/ee_states` (6,) no HDF5 oficial =
`concat(obs/ee_pos (3,), obs/ee_ori (3,))` exatamente (`np.array_equal`
testado). Concatenado com `obs/gripper_states` (2,) dá `state_dim=8`, batendo
com o `state_dim` já usado no config atual -- forte candidato a
reconstituir `observation.state` sem mudança de contrato.

**Consequência prática**: dado que a convenção é a mesma, os `norm_stats`
(`MinMaxNormalizer`) não precisam ser recalculados do zero por medo de
incompatibilidade de convenção -- mas ainda devem ser recalculados sobre o
HDF5 (min/max exatos podem diferir ligeiramente por serem um subconjunto de
episódios diferente do que o `lerobot/libero` usa, e por serem `float64`
onde o pipeline atual espera `float32` -- ver risco #4 do plano original).
Não há mais razão para suspeitar de corrupção silenciosa de aprendizado por
convenção de ação divergente.

### Metodologia -- leitura remota de HDF5 sem baixar o arquivo inteiro

Os arquivos oficiais têm ~700-800MB cada (maioria é imagem RGB
`128×128×3 uint8`, sem compressão). Para verificação, abrimos remotamente via
`fsspec.open(url, mode="rb")` + `h5py.File(f, "r")`, que faz HTTP range
requests sob demanda -- só os bytes das datasets pequenas (`actions`,
`ee_states` etc.) e da árvore de metadados HDF5 trafegam. Listar as chaves de
nível superior (`len(hf["data"].keys())`, contagem de demos) leva ~13-20s por
arquivo; inspecionar profundamente um demo inteiro (todas as obs + actions)
leva ~4min, porque a travessia da árvore de grupos/atributos do HDF5 sobre
HTTP soma muitas requisições pequenas. **Isso é só para verificação** -- a
implementação real do `HDF5LiberoDataset` deve baixar os arquivos para disco
(no Colab/Drive) e abrir localmente, não usar HTTP remoto em treino.

### Contagens de episódios por tarefa (para referência futura)

`lerobot/libero` tem 1693 episódios distribuídos de forma desigual entre as
40 tarefas (mín. 29, máx. 50 -- não é sempre os 50 demos completos de cada
arquivo oficial). Ao escrever o `HDF5LiberoDataset`, ou ele usa todos os
demos do arquivo oficial (mudando ligeiramente a quantidade de dado por
tarefa vs. o pipeline atual) ou replica exatamente o subconjunto de episódios
que o `lerobot/libero` usou -- **decisão em aberto, não resolvida aqui**;
afeta comparabilidade com os checkpoints já treinados no pipeline JPEG.

### 3. Imagens vêm invertidas verticalmente -- achado adicional (não estava no plano original)

Ao implementar `HDF5LiberoDataset`, uma checagem visual pixel-a-pixel revelou
que os arrays crus `obs/agentview_rgb` e `obs/eye_in_hand_rgb` do HDF5
oficial estão **invertidos verticalmente** em relação à convenção do vídeo
`lerobot/libero`:

- Frame cru do HDF5 (`agentview_rgb[0]` de `libero_object/pick_up_the_alphabet_soup_..._demo.hdf5`):
  braço do robô aparece "pendurado no teto", objetos flutuando -- fisicamente
  implausível pra uma cena de mesa.
- O mesmo frame com `arr[::-1]` (flip no eixo H): cena normal, braço saindo
  de trás/cima em direção à mesa, objetos apoiados -- fisicamente plausível.
- Confirmado contra o vídeo real: decodificado o frame 0 do episódio 0 de
  `lerobot/libero` (`videos/observation.images.image/chunk-000/file-000.mp4`,
  via `imageio`) -- a orientação bate exatamente com a versão invertida do
  HDF5, não com a crua.

Isso é consistente com a convenção conhecida do robosuite/MuJoCo (render
OpenGL com origem embaixo-esquerda, textura lida de cima pra baixo) --
`ImageModality`/utils do robomimic aplicam esse mesmo flip ao carregar HDF5
cru. `HDF5LiberoDataset._frame_to_tensor` aplica `arr[::-1]` em toda leitura
de câmera (`agentview_rgb` e `eye_in_hand_rgb`); testado em
`tests/test_hdf5_libero.py::test_flip_vertical_aplicado`.

**Sem esse flip, o treino aprenderia de imagens de cabeça para baixo,
silenciosamente** -- provavelmente sem erro, sem NaN, só um modelo pior (ou
que aprende a convenção errada e falha na hora do rollout, que usa a
convenção do env ao vivo -- ver `eval/obs_processing.py`, que já faz um flip
próprio, de 180°, mas para a fonte AO VIVO do simulador, um caso distinto
deste).

### 4. Implementação (passo 3 do plano) -- concluída

- `src/act_lang/data/hdf5_libero.py`: `HDF5LiberoDataset` (mesmo contrato de
  chaves/shapes de `PreprocessedLiberoDataset`), `load_task_mapping`,
  `load_meta`, `build_normalizers_from_meta`, `compute_stats`. Handles HDF5
  cacheados por `(path, os.getpid())`, abertos sob demanda em
  `__getitem__` -- nunca reabertos por amostra, nunca sobrevivem a um fork
  (metadados de comprimento lidos com `with h5py.File(...)` no `__init__`,
  fechados antes do `DataLoader` criar os workers).
- `scripts/download_libero_hdf5.py`: baixa os 40 arquivos de
  `configs/task_mapping_libero40.json` via `hf_hub_download` (retomável) e
  computa `meta.json` (stats min/max escaneando os arquivos baixados, NÃO
  copiados do lerobot -- ver seção 2 acima sobre por que recalcular).
- `scripts/train.py`: `build_data_hdf5(cfg, device, root)`, paralela a
  `build_data_preprocessed`; CLI `--hdf5-dir` (mutuamente exclusivo com
  `--preprocessed-dir`).
- `tests/test_hdf5_libero.py`: contrato de chaves/shapes, flip vertical,
  reconstituição de `observation.state`, padding de chunk na borda, filtro
  por tarefa/episódio, reuso de handle por processo, integração com
  `LiberoActBridge` -- HDF5 sintético pequeno em `tmp_path`, sem depender de
  download real.
- `pyproject.toml`: `h5py` adicionado a `dev` (pra rodar os testes) e a um
  extra novo `hdf5` (`h5py` + `huggingface_hub`, pro download real no Colab).

Ainda não executado: download real dos 40 arquivos (~28-32GB, requer
Colab/Drive) e o benchmark de validação (passo 6 do plano) e o treino piloto
(passo 7) -- ambos dependem desse download, que não foi feito aqui.

### Itens ainda em aberto

- Download real dos 40 HDF5 (rodar `scripts/download_libero_hdf5.py` no
  Colab) e benchmark de época de verdade (passo 6 do plano original) --
  ainda não medido, só estimado.
- Treino piloto curto comparando curva de loss com o pipeline JPEG (passo 7
  do plano original).
- Se replicar exatamente o subconjunto de episódios do `lerobot/libero` por
  tarefa, ou usar todos os demos disponíveis no HDF5 oficial (ver seção "1693
  episódios..." acima) -- `HDF5LiberoDataset` atualmente usa TODOS os demos
  disponíveis no arquivo oficial (tipicamente 50/tarefa), não o subconjunto
  do lerobot -- mais dado que o pipeline JPEG atual, o que muda a
  comparabilidade direta de curva de loss/época entre os dois pipelines.
