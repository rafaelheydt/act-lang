
## Backbone visual (ResNet18)

**Implementação**

```python
"""
Backbone visual (ResNet18) compartilhado entre câmeras.
"""

import torch
import torch.nn as nn
import torchvision


class VisionBackbone(nn.Module):
    def __init__(self, d_model: int, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        # Estatísticas que os pesos IMAGENET1K_V1 esperam na entrada.
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) em [0, 1] -> feature map (B, d_model, h, w)."""
        images = (images - self.img_mean) / self.img_std
        features = self.backbone(images)
        return self.proj(features)


def freeze_batchnorm(module: nn.Module) -> nn.Module:
    """Substitui todo BatchNorm2d por FrozenBatchNorm2d (padrão DETR/ACT original).

    Opcional, mas recomendado: BN com batches pouco diversos de imagens de robô
    degrada as estatísticas do pré-treino e cria gap treino/inferência.
    Chame ANTES de mover o modelo para o device e de criar o optimizer.
    """
    from torchvision.ops.misc import FrozenBatchNorm2d

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            frozen.weight.data.copy_(child.weight.data)
            frozen.bias.data.copy_(child.bias.data)
            frozen.running_mean.data.copy_(child.running_mean.data)
            frozen.running_var.data.copy_(child.running_var.data)
            setattr(module, name, frozen)
        else:
            freeze_batchnorm(child)
    return module
```

### Para que serve

Antes de qualquer coisa entrar no transformer, as imagens das câmeras precisam virar
uma sequência de "tokens" — vetores que o transformer sabe processar. É isso que a
`VisionBackbone` faz: recebe a imagem crua (`[0, 1]`) e devolve um mapa de features já em
`d_model` — pronto para ser achatado em tokens mais na frente.

Usa um **ResNet18 pré-treinado** (ImageNet) como extrator, cortando as duas últimas
camadas (*global average pooling* e a `fc` de classificação) — essas camadas colapsariam
a informação espacial num único vetor, e é justamente o mapa espacial que o transformer
precisa. Em seguida, um `Conv2d(512, d_model, kernel_size=1)` projeta os 512 canais do
ResNet para a dimensão que o resto do modelo usa (`d_model`) — a "conversão 1x1" é
essencialmente uma transformação linear aplicada a cada posição do mapa, independente.

### Normalização como buffer

```python
self.register_buffer("img_mean", ...)
self.register_buffer("img_std", ...)
...
images = (images - self.img_mean) / self.img_std
```

Os pesos pré-treinados do ResNet18 (`IMAGENET1K_V1`) esperam que a imagem de entrada
esteja normalizada pelas estatísticas do ImageNet — não os valores crus `[0, 1]`. Guardar
`img_mean`/`img_std` como **buffers** (em vez de, por exemplo, uma constante solta no
código) significa que eles viajam junto no `state_dict()` do módulo — ou seja, junto com
qualquer checkpoint salvo. Isso torna a normalização parte inseparável do modelo: não tem
como usar a `VisionBackbone` sem ela ser aplicada, não importa de onde a imagem venha.

### Frozen BatchNorm (opcional)

`freeze_batchnorm()` é uma função à parte, chamada explicitamente quando quiser — não
algo que acontece por padrão. Ela percorre o módulo recursivamente e substitui cada
`BatchNorm2d` por um `FrozenBatchNorm2d` (que usa estatísticas fixas, em vez de recalcular
média/variância a cada batch). Isso importa porque batches de imagens de robô costumam
ser pouco diversos (poucas cenas, pouca iluminação variando) — deixar o BatchNorm
"aprendendo" em cima disso degrada as estatísticas que vieram do pré-treino no ImageNet.

### Positional embedding: fora deste módulo

Repara que `forward()` devolve só o mapa de features (`(B, d_model, h, w)`) — nenhuma
informação de posição sai daqui. O cálculo e a aplicação do positional embedding 2D
acontecem em outro lugar, depois que esse mapa é achatado em tokens — ver **Bloco 4**.

---

## Bloco 4: Como a posição é aplicada no transformer

O bloco 3 (`VisionBackbone`) só devolve o mapa de features — nenhuma posição sai dali. É
logo depois, quando esse mapa vira uma sequência de tokens, que a posição entra em cena.
Como o `pos_embed` é usado de fato dentro do transformer é uma das decisões de design
mais importantes do ACT/DETR.

### `src` e `pos` viajam separados até o fim

Como já rastreamos, o caminho completo é: `Backbone` devolve `NestedTensor(x, pos_embs)`
→ `cvaeDecoderInputCollator` desempacota em `(src, pos, queries)` → `Transformer.forward`
recebe os três, ainda separados:

```python
class Transformer(DETRTransformer):
    def forward(self, src, pos_embed, query_embed):
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)
```

Em nenhum ponto até aqui alguém faz `src + pos_embed`. Essa soma só acontece **dentro**
de cada camada do encoder/decoder — código que não está no notebook, e sim no repositório
`facebookresearch/detr` clonado na célula 0 (`../detr/models/transformer.py`).

### Onde a soma de fato acontece: só em query/key, nunca em value

Dentro de `TransformerEncoderLayer`, existe um método auxiliar:

```python
def with_pos_embed(self, tensor, pos):
    return tensor if pos is None else tensor + pos
```

E ele é usado assim, no `forward_post` (caminho padrão do encoder):

```python
def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
    q = k = self.with_pos_embed(src, pos)
    src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    ...
```

Repara: `q` e `k` recebem `src + pos`, mas o `value=src` que entra na mesma chamada de
atenção é o conteúdo **original**, sem a posição somada. `TransformerDecoderLayer` segue o
mesmo padrão duas vezes — uma na self-attention entre as `action_queries` (`query_pos`
somado a `tgt`) e outra na cross-attention com a `memory` (`query_pos` em `tgt`, `pos` em
`memory`) — sempre só em `q`/`k`.

### Por que isso importa

- **Query e key decidem "quem presta atenção em quem"** — pra isso, saber a posição é
  essencial (ex: "esse pedaço de imagem fica perto daquele").
- **Value é o que de fato é agregado/copiado** pra saída — se a posição fosse somada
  aqui também, o conteúdo real ficaria permanentemente misturado com um sinal de posição,
  e nenhuma camada seguinte conseguiria mais separar os dois.
- **A soma é refeita em toda camada**, com o mesmo `pos` original — não é acumulativa.
  Isso significa que mesmo a última camada do encoder/decoder ainda tem acesso à posição
  "pura", nunca borrada por camadas anteriores.

💡 Insight: se o `pos_embed` fosse somado uma única vez, lá no início (dentro da própria
`Backbone`, por exemplo), o efeito seria parecido no output final, mas cada camada
subsequente só teria acesso a uma versão cada vez mais "misturada" de posição+conteúdo,
sem o sinal de posição isolado disponível pra recalcular a atenção do zero. Reinjetar em
toda camada é o que garante que a posição continue "limpa" e disponível fim a fim.