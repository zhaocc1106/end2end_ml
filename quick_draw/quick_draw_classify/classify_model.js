/**
 * @file Train quick draw classifier model.
 * @author zhaochaochao@baidu.com
 * @date 2019/12/6
 */
const tf = require('@tensorflow/tfjs-node-gpu');

const BATCH_SIZE = 10; // The mini batch size.
const DROPOUT_RATE = 0.3; // The dropout rate if have.
const CNN_LEN = [5, 5, 3]; // The cnn layers kernel size.
const CNN_FILTERS = [48, 64, 96]; // The cnn layers filters number.
const RNN_TYPE = "lstm"; // The rnn layer type.
const RNN_LAYERS = 3; // The rnn layers number.
const RNN_UNITS = 128; // The rnn layer units number.

/**
 * The reduce sum layer.
 */
class ReduceSumLayer extends tf.layers.Layer {
    constructor(axis) {
        super({});
        this.axis = axis;
    }

    // In this case, the output is a scalar.
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[2]];
    }

    // call() is where we do the computation.
    call(input, kwargs) {
        return tf.sum(input, this.axis, false);
    }

    // Every layer needs a unique name.
    getClassName() {
        return 'ReduceSum';
    }
}

/**
 * Build the model with functional api.
 *
 * @param {number[]} cnnLen: Length of the convolution filters.
 * @param {number[]} cnnFilters: The number of the convolution filters.
 * @param {number} rnnLayers: The rnn layers number.
 * @param {number} units: The rnn units number.
 * @param {string} rnnType: The rnn type. "lstm" or "gru".
 * @param {boolean} batchNorm: If batch normalization.
 * @param {boolean} dropout: If dropout.
 * @param {number} dropoutRate: Dropout rate.
 * @param {number} batchSize: The batch size.
 */
function modelFunc(cnnLen = CNN_LEN, cnnFilters = CNN_FILTERS,
                   rnnLayers = RNN_LAYERS, units = RNN_UNITS, rnnType = RNN_TYPE,
                   batchNorm = true,
                   dropout = true, dropoutRate = DROPOUT_RATE,
                   batchSize = BATCH_SIZE) {
    // The inks input. (x_delta, y_delta)
    let inksInp = tf.layers.input({batchShape: [batchSize, null, 2]});
    // The length input of inks.
    let lenInp = tf.layers.input({batchShape: [batchSize]});

    // Add convolution layers.
    let con = inksInp;
    for (let i = 0; i < CNN_LEN.length; ++i) {
        con = tf.layers.conv1d({
            kernelSize: cnnLen[i],
            filters: cnnFilters[i],
            padding: "same",
            strides: 1
        }).apply(con);
        if (batchNorm) {
            con = tf.layers.batchNormalization().apply(con);
        }
        if (dropout) {
            con = tf.layers.dropout(dropoutRate).apply(con);
        }
    }

    // Add mask layers.
    let mask = tf.layers.masking({maskValue: 0}).apply(con);

    // Add rnn layers.
    let rnn = mask;
    let cell = null;
    if (rnnType === "lstm") {
        cell = tf.layers.lstm;
    } else if (rnnType === "gru") {
        cell = tf.layers.gru;
    } else {
        console.trace("wrong rnn type.");
        return null;
    }
    for (let i = 0; i < rnnLayers; ++i) {
        let rnnLayer = cell({
            units: units,
            returnSequences: i !== rnnLayers - 1,
            activation: "sigmoid",
            recurrentInitializer: "glorotUniform",
            goBackwards: false,
            dropout: dropoutRate
        });
        rnn = tf.layers.bidirectional({layer: rnnLayer, mergeMode: "concat"}).apply(rnn);
    }

    // Reduce sum.
    // reduceSum = new ReduceSumLayer(1).apply(rnn);

    // Add dense layers.
    logits = tf.layers.dense({units: 1000, activation: "softmax"}).apply(rnn);

    return {"inp": [inksInp, lenInp], "output": logits};
}

const inpAndOut = modelFunc();
console.log(inpAndOut);
const model = tf.model({inputs: inpAndOut["inp"], outputs: inpAndOut["output"]});
model.summary();
model.save('file:///tmp/quick_draw_classify').then(() => {
    console.log("save complete.");
});