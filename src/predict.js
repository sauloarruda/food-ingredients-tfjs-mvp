const fs = require('node:fs');
const path = require('node:path');
const tf = require('./tf');
const { buildModel } = require('./model');
const { imageToTensor3d } = require('./dataLoader');

const DEFAULT_THRESHOLD = Number(process.env.THRESHOLD || 0.5);
const DEFAULT_TOP_K = Number(process.env.TOP_K || 5);

/**
 * Converte uma imagem em tensor 4D com batch size 1 para inferencia.
 *
 * Entrada:
 * - imagePath: caminho da imagem.
 *
 * Saida:
 * - Tensor [1, 224, 224, 3].
 */
async function imageToTensor(imagePath) {
  const image = await imageToTensor3d(imagePath);

  // expandDims adiciona a dimensao de batch exigida pelo model.predict.
  const input = image.expandDims(0);

  image.dispose();
  return input;
}

/**
 * Carrega a lista de ingredientes usada pelo modelo salvo.
 *
 * Prioridade:
 * - saved_model/ingredients.json, que garante alinhamento com os pesos.
 * - dataset/dataset.json, apenas como fallback.
 *
 * Saida:
 * - Array de ingredientes/classes.
 */
function loadSavedIngredients() {
  const ingredientsPath = path.resolve(__dirname, '..', 'saved_model', 'ingredients.json');
  const manifestPath = path.resolve(__dirname, '..', 'dataset', 'dataset.json');

  if (fs.existsSync(ingredientsPath)) {
    return JSON.parse(fs.readFileSync(ingredientsPath, 'utf8'));
  }

  if (fs.existsSync(manifestPath)) {
    const items = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
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

  return ['tomate', 'arroz', 'carne'];
}

/**
 * Carrega o modelo salvo, ou cria um modelo aleatorio se ainda nao houver
 * modelo treinado.
 *
 * Saida:
 * - Objeto com { model, ingredients }.
 */
async function loadOrCreateModel() {
  const modelPath = path.resolve(__dirname, '..', 'saved_model', 'model.json');
  const ingredients = loadSavedIngredients();

  if (fs.existsSync(modelPath)) {
    return {
      // tf.loadLayersModel carrega um modelo salvo por model.save no formato
      // Layers do TensorFlow.js.
      model: await tf.loadLayersModel(`file://${modelPath}`),
      ingredients
    };
  }

  console.warn('Modelo salvo não encontrado. Usando modelo inicializado aleatoriamente.');
  return {
    model: buildModel(ingredients.length),
    ingredients
  };
}

/**
 * Executa inferencia multi-label em uma imagem.
 *
 * Entrada:
 * - imagePath: caminho da imagem.
 * - options.threshold: score minimo para aceitar um ingrediente.
 * - options.topK: quantidade de candidatos retornados se nada passar do
 *   threshold.
 *
 * Saida:
 * - Lista ordenada de ingredientes com score.
 */
async function predict(imagePath, options = {}) {
  const threshold = options.threshold ?? DEFAULT_THRESHOLD;
  const topK = options.topK ?? DEFAULT_TOP_K;

  if (!imagePath) {
    throw new Error('Informe o caminho da imagem. Exemplo: npm run predict -- ./comida.jpg');
  }

  if (!fs.existsSync(imagePath)) {
    throw new Error(`Imagem nao encontrada: ${imagePath}`);
  }

  const { model, ingredients } = await loadOrCreateModel();
  const input = await imageToTensor(imagePath);

  // model.predict executa a passada de inferencia e retorna um tensor de scores
  // sigmoid, um score por ingrediente.
  const prediction = model.predict(input);

  // data copia os scores do tensor para um array JavaScript tipado.
  const scores = Array.from(await prediction.data());

  const rankedIngredients = scores
    .map((score, index) => ({
      ingredient: ingredients[index],
      score
    }))
    .sort((a, b) => b.score - a.score);

  const predictedIngredients = rankedIngredients.filter((item) => item.score > threshold);
  const result = predictedIngredients.length > 0
    ? predictedIngredients
    : rankedIngredients.slice(0, topK).map((item) => ({
        ...item,
        belowThreshold: true
      }));

  input.dispose();
  prediction.dispose();
  model.dispose();

  return result;
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
