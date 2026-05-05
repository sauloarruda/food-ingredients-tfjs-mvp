const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const tf = require('./tf');
const { buildModel } = require('./model');
const { INGREDIENTS } = require('./dataLoader');

const IMAGE_SIZE = 224;
const THRESHOLD = 0.5;

async function imageToTensor(imagePath) {
  const buffer = await sharp(imagePath)
    .resize(IMAGE_SIZE, IMAGE_SIZE)
    .removeAlpha()
    .raw()
    .toBuffer();

  return tf.tidy(() => {
    const image = tf.tensor3d(new Uint8Array(buffer), [IMAGE_SIZE, IMAGE_SIZE, 3]);
    return image.toFloat().div(255).expandDims(0);
  });
}

async function loadOrCreateModel() {
  const modelPath = path.resolve(__dirname, '..', 'saved_model', 'model.json');

  if (fs.existsSync(modelPath)) {
    return tf.loadLayersModel(`file://${modelPath}`);
  }

  console.warn('Modelo salvo nao encontrado. Usando modelo inicializado aleatoriamente.');
  return buildModel(INGREDIENTS.length);
}

async function predict(imagePath) {
  if (!imagePath) {
    throw new Error('Informe o caminho da imagem. Exemplo: npm run predict -- ./comida.jpg');
  }

  if (!fs.existsSync(imagePath)) {
    throw new Error(`Imagem nao encontrada: ${imagePath}`);
  }

  const model = await loadOrCreateModel();
  const input = await imageToTensor(imagePath);
  const prediction = model.predict(input);
  const scores = Array.from(await prediction.data());

  const ingredients = scores
    .map((score, index) => ({
      ingredient: INGREDIENTS[index],
      score
    }))
    .filter((item) => item.score > THRESHOLD)
    .sort((a, b) => b.score - a.score);

  input.dispose();
  prediction.dispose();
  model.dispose();

  return ingredients;
}

if (require.main === module) {
  const imagePath = process.argv[2];

  predict(imagePath)
    .then((ingredients) => {
      console.log(JSON.stringify({ ingredients }, null, 2));
    })
    .catch((error) => {
      console.error('Erro ao predizer ingredientes:', error.message);
      process.exit(1);
    });
}

module.exports = {
  predict,
  imageToTensor
};
