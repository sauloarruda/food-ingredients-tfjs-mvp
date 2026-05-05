const util = require('node:util');

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

module.exports = require('@tensorflow/tfjs-node');
