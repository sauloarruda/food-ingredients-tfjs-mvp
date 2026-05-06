const util = require('node:util');

/**
 * Shim de compatibilidade para Node 22.
 *
 * O pacote @tensorflow/tfjs-node 4.22.0 ainda acessa helpers antigos do modulo
 * node:util. Definimos substitutos antes de carregar o TensorFlow para evitar
 * erro ou deprecation warning em runtime.
 */
Object.defineProperties(util, {
  isNullOrUndefined: {
    configurable: true,
    enumerable: false,
    value: (value) => value === null || value === undefined
  },
  isArray: {
    configurable: true,
    enumerable: false,
    value: Array.isArray
  }
});

// Exporta sempre o TensorFlow.js Node a partir deste arquivo. Os outros
// arquivos do projeto devem importar './tf', nao '@tensorflow/tfjs-node'
// diretamente, para garantir que o shim acima seja aplicado primeiro.
module.exports = require('@tensorflow/tfjs-node');
