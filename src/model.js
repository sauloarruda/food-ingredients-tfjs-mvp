const tf = require('./tf');

/**
 * Cria uma loss binary crossentropy ponderada para cenarios multi-label
 * desbalanceados, onde existem muito mais labels negativos do que positivos.
 *
 * Entrada:
 * - positiveWeight: multiplicador aplicado apenas aos erros dos labels 1.
 *
 * Saida:
 * - Funcao de loss compativel com model.compile.
 */
function weightedBinaryCrossentropy(positiveWeight) {
  // tf.tidy executa a loss e descarta tensores intermediarios criados em cada
  // batch, reduzindo vazamento de memoria durante o treino.
  return (yTrue, yPred) => tf.tidy(() => {
    // tf.scalar cria um tensor escalar com valor 1 para operacoes tensorais.
    const one = tf.scalar(1);

    // tf.scalar cria o peso positivo como tensor para permitir broadcast
    // contra os tensores de labels e predicoes.
    const weight = tf.scalar(positiveWeight);

    // clipByValue limita as predicoes para evitar log(0), que geraria
    // infinito e quebraria a loss.
    const clipped = yPred.clipByValue(1e-7, 1 - 1e-7);

    // mul aplica a parte positiva da binary crossentropy e multiplica pelo
    // peso configurado para reduzir o efeito do desbalanceamento.
    const positiveLoss = yTrue.mul(clipped.log()).mul(weight);

    // sub calcula o complemento dos labels/predicoes, e log aplica a parte
    // negativa da binary crossentropy.
    const negativeLoss = one.sub(yTrue).mul(one.sub(clipped).log());

    // add combina perdas positiva e negativa; mean agrega em um unico valor;
    // neg troca o sinal porque a formula usa logs negativos.
    return positiveLoss.add(negativeLoss).mean().neg();
  });
}

/**
 * Compila um modelo TensorFlow.js com otimizador, loss e metricas.
 *
 * Entrada:
 * - model: modelo de camadas criado pelo TensorFlow.js.
 * - loss: nome de loss ou funcao customizada.
 */
function compileModel(model, loss = 'binaryCrossentropy') {
  // model.compile prepara o modelo para treino definindo otimizador, funcao de
  // perda e metricas monitoradas.
  model.compile({
    // tf.train.adam cria o otimizador Adam com learning rate fixo.
    optimizer: tf.train.adam(0.001),
    loss,
    metrics: ['accuracy']
  });
}

/**
 * Cria o modelo sequencial minimo do MVP.
 *
 * Entrada:
 * - numClasses: quantidade de ingredientes/classes na saida.
 * - options.positiveWeight: peso opcional para loss ponderada.
 *
 * Saida:
 * - Modelo TensorFlow.js compilado.
 */
function buildModel(numClasses, options = {}) {
  // tf.sequential cria um modelo linear, onde cada camada alimenta a proxima.
  const model = tf.sequential();

  // tf.layers.flatten transforma a imagem [224, 224, 3] em um vetor 1D para
  // alimentar camadas densas simples.
  model.add(tf.layers.flatten({ inputShape: [224, 224, 3] }));

  // tf.layers.dense cria uma camada totalmente conectada com ativacao ReLU.
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));

  // tf.layers.dense cria a camada de saida multi-label; sigmoid gera um score
  // independente entre 0 e 1 para cada ingrediente.
  model.add(tf.layers.dense({ units: numClasses, activation: 'sigmoid' }));

  const loss = options.positiveWeight
    ? weightedBinaryCrossentropy(options.positiveWeight)
    : 'binaryCrossentropy';

  compileModel(model, loss);

  return model;
}

module.exports = {
  buildModel,
  compileModel
};
