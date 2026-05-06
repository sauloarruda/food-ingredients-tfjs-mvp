const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const tf = require('./tf');

const IMAGE_SIZE = 224;

/**
 * Carrega uma imagem do disco e converte para um tensor 3D normalizado.
 *
 * Entrada:
 * - imagePath: caminho absoluto ou relativo para uma imagem.
 *
 * Saida:
 * - Tensor com shape [224, 224, 3], canais RGB e valores entre 0 e 1.
 */
async function imageToTensor3d(imagePath) {
  const buffer = await sharp(imagePath)
    .resize(IMAGE_SIZE, IMAGE_SIZE, { fit: 'cover' })
    .removeAlpha()
    .toColorspace('srgb')
    .raw()
    .toBuffer();

  // tf.tidy executa as operacoes internas e libera tensores temporarios
  // automaticamente, evitando vazamento de memoria entre imagens.
  return tf.tidy(() => {
    // tf.tensor3d cria um tensor de imagem com altura, largura e canais RGB
    // a partir dos bytes crus retornados pelo sharp.
    const image = tf.tensor3d(new Uint8Array(buffer), [IMAGE_SIZE, IMAGE_SIZE, 3]);

    // toFloat converte os pixels inteiros, originalmente 0 a 255, para
    // float32, que e o tipo esperado pelo modelo.
    const floatImage = image.toFloat();

    // div(255) normaliza os pixels para a faixa 0 a 1, reduzindo escala dos
    // valores de entrada e deixando o treino mais estavel.
    return floatImage.div(255);
  });
}

/**
 * Le e valida o arquivo JSON que descreve o dataset.
 *
 * Entrada:
 * - manifestPath: caminho para dataset.json.
 *
 * Saida:
 * - Array de itens no formato { image, ingredients }.
 */
function loadDatasetManifest(manifestPath) {
  const raw = fs.readFileSync(manifestPath, 'utf8');
  const items = JSON.parse(raw);

  if (!Array.isArray(items)) {
    throw new Error('O arquivo de dataset precisa ser um array de itens.');
  }

  return items;
}

/**
 * Extrai a lista unica de ingredientes preservando a ordem em que aparecem
 * no dataset. Essa ordem define o indice de cada classe nos labels multi-hot.
 *
 * Entrada:
 * - items: array carregado do manifesto.
 *
 * Saida:
 * - Array de strings com os ingredientes/classes.
 */
function getIngredientsFromItems(items) {
  const ingredients = [];
  const seen = new Set();

  for (const item of items) {
    for (const ingredient of item.ingredients || []) {
      if (!seen.has(ingredient)) {
        seen.add(ingredient);
        ingredients.push(ingredient);
      }
    }
  }

  return ingredients;
}

/**
 * Converte a lista de ingredientes de uma imagem em um vetor multi-hot.
 *
 * Exemplo:
 * - itemIngredients: ['arroz', 'carne']
 * - ingredients: ['arroz', 'feijao', 'carne']
 * - retorno: [1, 0, 1]
 *
 * Entrada:
 * - itemIngredients: ingredientes presentes em uma imagem.
 * - ingredients: lista completa de classes do dataset.
 *
 * Saida:
 * - Array numerico com 1 para ingrediente presente e 0 para ausente.
 */
function createMultiHotLabel(itemIngredients, ingredients) {
  const label = new Array(ingredients.length).fill(0);

  for (const ingredient of itemIngredients || []) {
    const index = ingredients.indexOf(ingredient);
    if (index >= 0) {
      label[index] = 1;
    }
  }

  return label;
}

/**
 * Carrega o dataset completo em memoria para treino.
 *
 * Entrada:
 * - manifestPath: caminho para dataset.json.
 * - imagesDir: pasta onde as imagens referenciadas pelo manifesto estao.
 *
 * Saida:
 * - images: tensor 4D [samples, 224, 224, 3].
 * - labels: tensor 2D [samples, numClasses].
 * - ingredients: lista de classes usada para mapear os indices.
 */
async function loadImageDataset(manifestPath, imagesDir) {
  const items = loadDatasetManifest(manifestPath);
  const ingredients = getIngredientsFromItems(items);

  if (items.length === 0) {
    throw new Error('Dataset vazio.');
  }

  if (ingredients.length === 0) {
    throw new Error('Nenhum ingrediente encontrado no dataset.');
  }

  const imageTensors = [];
  const labelsData = [];

  for (const item of items) {
    if (!item.image) {
      throw new Error('Item do dataset sem campo "image".');
    }

    const imagePath = path.resolve(imagesDir, item.image);

    if (!fs.existsSync(imagePath)) {
      throw new Error(`Imagem nao encontrada: ${imagePath}`);
    }

    imageTensors.push(await imageToTensor3d(imagePath));
    labelsData.push(createMultiHotLabel(item.ingredients, ingredients));
  }

  // tf.stack combina a lista de tensores 3D de cada imagem em um unico
  // tensor 4D no formato esperado pelo model.fit.
  const images = tf.stack(imageTensors);

  // tf.tensor2d converte a matriz JavaScript de labels multi-hot em tensor
  // 2D, com uma linha por imagem e uma coluna por ingrediente/classe.
  const labels = tf.tensor2d(labelsData, [items.length, ingredients.length]);

  imageTensors.forEach((tensor) => tensor.dispose());

  return {
    images,
    labels,
    ingredients
  };
}

module.exports = {
  IMAGE_SIZE,
  loadImageDataset,
  imageToTensor3d
};
