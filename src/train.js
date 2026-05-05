const path = require('node:path');
const tf = require('./tf');
const { buildModel } = require('./model');
const { INGREDIENTS, createFakeDataset } = require('./dataLoader');

async function train() {
  const model = buildModel(INGREDIENTS.length);
  const { images, labels } = createFakeDataset(32, INGREDIENTS.length);

  await model.fit(images, labels, {
    epochs: 3,
    batchSize: 8,
    validationSplit: 0.2,
    shuffle: true
  });

  const modelDir = path.resolve(__dirname, '..', 'saved_model');
  await model.save(`file://${modelDir}`);

  images.dispose();
  labels.dispose();
  model.dispose();
  tf.disposeVariables();

  console.log(`Modelo salvo em: ${modelDir}`);
}

train().catch((error) => {
  console.error('Erro ao treinar modelo:', error);
  process.exit(1);
});
