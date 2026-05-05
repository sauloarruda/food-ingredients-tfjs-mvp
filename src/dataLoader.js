const tf = require('./tf');

const INGREDIENTS = [
  'tomate',
  'queijo',
  'alface',
  'carne',
  'arroz',
  'frango',
  'cebola',
  'massa'
];

function createFakeDataset(samples = 32, numClasses = INGREDIENTS.length) {
  const images = tf.randomUniform([samples, 224, 224, 3]);
  const labelsData = Array.from({ length: samples }, () =>
    Array.from({ length: numClasses }, () => (Math.random() > 0.65 ? 1 : 0))
  );
  const labels = tf.tensor2d(labelsData, [samples, numClasses]);

  return {
    images,
    labels
  };
}

module.exports = {
  INGREDIENTS,
  createFakeDataset
};
