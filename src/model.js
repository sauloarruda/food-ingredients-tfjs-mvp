const tf = require('./tf');

function buildModel(numClasses) {
  const model = tf.sequential();

  model.add(tf.layers.flatten({ inputShape: [224, 224, 3] }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

module.exports = {
  buildModel
};
