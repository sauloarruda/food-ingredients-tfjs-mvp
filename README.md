# Food Ingredients TFJS MVP

MVP em Node.js para reconhecimento multi-label de ingredientes em imagens de comida usando `@tensorflow/tfjs-node`.

O objetivo deste projeto é validar o pipeline completo:

- carregar imagens locais;
- carregar labels multi-label de um JSON;
- treinar um modelo simples;
- salvar modelo e lista de classes;
- predizer ingredientes de uma imagem.

Este MVP não usa APIs externas e não usa GPU.

## Requisitos

- Node.js
- npm

Dependências principais:

- `@tensorflow/tfjs-node`
- `sharp`

## Instalação

```bash
npm install
```

### Dataset

O dataset é mantido em um repositório privado separado para proteger os dados. Para acessar o dataset, você precisa ser um colaborador do repositório privado.

Clone o projeto com submódulos:

```bash
git clone --recursive https://github.com/sauloarruda/food-ingredients-tfjs-mvp.git
```

Se você já clonou sem submódulos, inicialize-os:

```bash
git submodule update --init --recursive
```


Se você precisar de um dataset para seus testes, crie uma issue no projeto com as seguintes informações:

- Seu nome (se já não estiver presente)
- Email
- Telefone de contato
- Empresa (se aplicável)
- Objetivo do uso

Ao solicitar acesso, você está ciente e de acordo com os termos da licença em [LICENSE](LICENSE).

## Estrutura

```text
src/
  dataLoader.js
  model.js
  train.js
  predict.js
  tf.js

dataset/  # Submódulo privado
  dataset.json
  images/

saved_model/ é gerado pelo treino e fica fora do Git.

dataset/ é o diretório do dataset. Está configurado como um submódulo Git apontando para o repositório privado do dataset. 
```

## Formato Do Dataset

O arquivo padrão é `dataset/dataset.json`.

Formato esperado:

```json
[
  {
    "image": "foto.jpg",
    "ingredients": ["arroz", "carne", "cebola"]
  }
]
```

As imagens devem estar em:

```text
dataset/images/
```

A lista final de classes é derivada automaticamente dos ingredientes presentes no JSON, preservando a ordem de primeira aparição.

## Treinar

```bash
npm run train
```

Por padrão, o script usa:

```text
dataset/dataset.json
dataset/images/
```

Também é possível informar caminhos manualmente:

```bash
node src/train.js ./dataset/dataset.json ./dataset/images
```

Variáveis úteis:

```bash
EPOCHS=80 BATCH_SIZE=1 VALIDATION_SPLIT=0 npm run train
```

O treino salva:

```text
saved_model/model.json
saved_model/weights.bin
saved_model/ingredients.json
```

## Predizer

```bash
npm run predict -- dataset/images/alguma-imagem.jpg
```

Exemplo:

```bash
npm run predict -- dataset/images/20260425_084559.jpg
```

Variáveis úteis:

```bash
THRESHOLD=0.4 TOP_K=10 npm run predict -- dataset/images/20260425_084559.jpg
```

Se nenhum ingrediente passar do `THRESHOLD`, o script retorna os `TOP_K` maiores scores com `belowThreshold: true`. Isso ajuda no MVP porque o dataset ainda é pequeno e os scores podem ficar baixos.

## Observações Importantes

O modelo atual é propositalmente simples:

- `Flatten`
- `Dense`
- saida `sigmoid`

Com poucas imagens e muitas classes, ele tende a aprender ingredientes frequentes e pode não acertar imagens do próprio dataset. Isso é esperado nesta fase.

Para melhorar qualidade:

- aumentar bastante o número de imagens;
- reduzir/normalizar labels muito específicos;
- agrupar sinônimos e variantes;
- usar arquitetura convolucional leve;
- separar treino/validação apenas quando houver dados suficientes;
- avaliar top-k além de threshold fixo.

## Auditoria

Foi usado `overrides` em `package.json` para atualizar `tar` transitivo e zerar vulnerabilidades conhecidas reportadas pelo `npm audit`.

```bash
npm audit
```

## Node 22

`src/tf.js` contém um shim pequeno para compatibilidade com `@tensorflow/tfjs-node` em Node 22, evitando problemas com helpers depreciados de `node:util`.
