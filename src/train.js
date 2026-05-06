const fs = require('node:fs');
const path = require('node:path');
const tf = require('./tf');
const { buildModel, compileModel } = require('./model');
const { loadImageDataset } = require('./dataLoader');

/**
 * Calcula um peso para labels positivos com base no desbalanceamento do
 * dataset multi-label.
 *
 * Entrada:
 * - labels: tensor 2D [samples, numClasses] com labels multi-hot.
 *
 * Saida:
 * - Numero usado para aumentar o peso dos labels positivos na loss.
 */
async function calculatePositiveWeight(labels) {
  // labels.sum soma todos os valores 1 no tensor de labels, indicando quantos
  // exemplos positivos existem no dataset.
  // data copia o valor do tensor para JavaScript.
  const positiveCount = (await labels.sum().data())[0];
  const totalCount = labels.size;
  const negativeCount = totalCount - positiveCount;

  if (positiveCount === 0) {
    return 1;
  }

  return Math.min(20, Math.max(1, negativeCount / positiveCount));
}

/**
 * Executa o fluxo completo de treino:
 * - resolve caminhos do dataset;
 * - carrega imagens e labels;
 * - cria/treina o modelo;
 * - salva modelo e classes.
 */
async function train() {
  const manifestPath = path.resolve(process.argv[2] || path.join('dataset', 'dataset.json'));
  const imagesDir = path.resolve(process.argv[3] || path.join(path.dirname(manifestPath), 'images'));

  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Dataset nao encontrado: ${manifestPath}`);
  }

  const dataset = await loadImageDataset(manifestPath, imagesDir);
  const { images, labels, ingredients } = dataset;
  const positiveWeight = Number(process.env.POSITIVE_WEIGHT || await calculatePositiveWeight(labels));
  const model = buildModel(ingredients.length, { positiveWeight });
  const validationSplit = process.env.VALIDATION_SPLIT
    ? Number(process.env.VALIDATION_SPLIT)
    : images.shape[0] >= 20 ? 0.2 : 0;

  // model.fit treina o modelo usando os tensores de imagens e labels.
  await model.fit(images, labels, {
    epochs: Number(process.env.EPOCHS || 10),
    batchSize: Number(process.env.BATCH_SIZE || 4),
    validationSplit,
    shuffle: true
  });

  const modelDir = path.resolve(__dirname, '..', 'saved_model');

  // Recompila o modelo com loss padrao antes de salvar, evitando serializar
  // uma loss customizada que pode dificultar o carregamento na inferencia.
  compileModel(model);

  // model.save grava arquitetura e pesos no formato TensorFlow.js Layers.
  await model.save(`file://${modelDir}`);
  fs.writeFileSync(
    path.join(modelDir, 'ingredients.json'),
    JSON.stringify(ingredients, null, 2)
  );

  images.dispose();
  labels.dispose();
  model.dispose();

  // tf.disposeVariables libera variaveis TensorFlow mantidas globalmente pelo
  // backend apos o treino.
  tf.disposeVariables();

  console.log(`Modelo salvo em: ${modelDir}`);
  console.log(`Classes treinadas: ${ingredients.length}`);
  console.log(`Peso positivo usado: ${positiveWeight.toFixed(2)}`);
}

train().catch((error) => {
  console.error('Erro ao treinar modelo:', error);
  process.exit(1);
});
