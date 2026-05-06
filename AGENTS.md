# AGENTS.md

Instruções para agentes de AI trabalhando neste repositório.

## Contexto Do Projeto

Este é um MVP de reconhecimento multi-label de ingredientes em imagens de comida.

Stack:

- Node.js
- `@tensorflow/tfjs-node`
- `sharp`

Não usar:

- APIs externas;
- GPU;
- modelos pesados nesta fase.

O objetivo atual é validar o pipeline local de treino e inferência, não obter alta acurácia.

## Arquivos Principais

- `src/dataLoader.js`: carrega `dataset/dataset.json`, lê imagens com `sharp`, transforma imagens em tensores e labels em multi-hot.
- `src/model.js`: define e compila o modelo TensorFlow.js.
- `src/train.js`: treina com dataset real e salva `saved_model/`.
- `src/predict.js`: carrega modelo salvo, processa imagem e retorna ingredientes.
- `src/tf.js`: shim de compatibilidade para Node 22 antes de importar `@tensorflow/tfjs-node`.
- `dataset/dataset.json`: manifesto com imagens e ingredientes.
- `dataset/images/`: imagens locais.

## Regras De Desenvolvimento

- Manter o código simples e orientado a MVP.
- Preferir mudanças pequenas e verificáveis.
- Não reintroduzir dataset fake silencioso no treino. Se o dataset real não existir, o treino deve falhar com erro claro.
- Não hardcodar a lista de ingredientes no código principal. As classes devem vir do dataset ou de `saved_model/ingredients.json`.
- Não commitar `node_modules/`, `saved_model/`, `.DS_Store` ou imagens temporárias.
- Preservar o uso de `sharp` para preprocessamento de imagem.
- Preservar consistência entre preprocessamento de treino e predição.

## Dataset

Formato esperado:

```json
[
  {
    "image": "arquivo.jpg",
    "ingredients": ["arroz", "carne"]
  }
]
```

As imagens ficam em `dataset/images/`.

As classes são derivadas automaticamente da ordem de primeira aparição dos ingredientes no JSON. Essa ordem precisa bater com `saved_model/ingredients.json`.

## Comandos De Verificação

Depois de alterar código:

```bash
node --check src/dataLoader.js
node --check src/model.js
node --check src/train.js
node --check src/predict.js
```

Treino:

```bash
npm run train
```

Predição:

```bash
npm run predict -- dataset/images/20260425_084559.jpg
```

Audit:

```bash
npm audit
```

## Variáveis De Ambiente

Treino:

- `EPOCHS`: número de épocas.
- `BATCH_SIZE`: tamanho do batch.
- `VALIDATION_SPLIT`: fração do dataset usada para validação.
- `POSITIVE_WEIGHT`: peso manual para labels positivos na loss, se usado.

Predição:

- `THRESHOLD`: score mínimo para retornar ingrediente.
- `TOP_K`: quantidade de candidatos retornados quando nada passa do threshold.

## Estado Técnico Atual

O pipeline de inferência funciona, mas a qualidade do modelo ainda é limitada pelo dataset pequeno e pelas muitas classes. Em testes com imagens do próprio dataset, os scores podem ficar baixos ou priorizar ingredientes frequentes.

Se for melhorar qualidade, priorizar:

1. normalização de labels;
2. aumento do dataset;
3. arquitetura convolucional leve;
4. métricas top-k para debug;
5. tratamento de desbalanceamento multi-label.

## Cuidado Com TensorFlow.js

`@tensorflow/tfjs-node` pode ter incompatibilidades em Node 22. Sempre importar TensorFlow via:

```js
const tf = require('./tf');
```

Não importar `@tensorflow/tfjs-node` diretamente nos arquivos do projeto.
