import numpy as np


class Optimizer:
    @staticmethod
    def gradientDecent(this, learningRate=0.01):
        learningRate = np.float32(learningRate)

        def optimizer(layer):
            this.deltaBiases[layer] *= learningRate
            this.deltaWeights[layer] *= learningRate

        return optimizer

    @staticmethod
    def momentum(this, learningRate=0.01, alpha=None):
        if alpha is None:
            alpha = learningRate / 10
        learningRate = np.float32(learningRate)
        ALPHA = np.float32(alpha)
        this.pdb = list(range(this.numLayers))  # pdb -> prev_delta_biases
        this.pdw = this.pdb.copy()  # pdw -> prev_delta_weights

        def optimizer(layer):
            this.deltaBiases[layer] = this.pdb[layer] = \
                ALPHA * this.pdb[layer] + learningRate * this.deltaBiases[layer]
            this.deltaWeights[layer] = this.pdw[layer] = \
                ALPHA * this.pdw[layer] + learningRate * this.deltaWeights[layer]

        return optimizer

    # re-implement
    @staticmethod
    def nesterov(this, learningRate=0.01, alpha=None):
        raise Warning("nesterov is not fully implemented, use momentum instead")
        if alpha is None: alpha = learningRate
        learningRate = np.float32(learningRate)
        ALPHA = np.float32(alpha)
        this.pdb = list(range(this.numLayers))  # pdb -> prev_delta_biases
        this.pdw = this.pdb.copy()  # pdw -> prev_delta_weights

        def optimizer(layer):
            this.theta_weights[layer] = this.weightsList[layer] - ALPHA * this.pdw[layer]
            this.theta_biases[layer] = this.biasesList[layer] - ALPHA * this.pdb[layer]

            this.deltaBiases[layer] = this.pdb[layer] = \
                ALPHA * this.pdb[layer] + learningRate * this.deltaBiases[layer]
            this.deltaWeights[layer] = this.pdw[layer] = \
                ALPHA * this.pdw[layer] + learningRate * this.deltaWeights[layer]

        return optimizer

    @staticmethod
    def decay(this, learningRate=0.01, alpha=None):
        if alpha is None:
            alpha = 1 / learningRate
        learningRate = np.float32(learningRate)
        ALPHA = np.float32(alpha)
        this.decayCount = 0

        def optimizer(layer):
            k = learningRate / (1 + this.decayCount / ALPHA)
            this.deltaBiases[layer] *= k
            this.deltaWeights[layer] *= k

            this.decayCount += 1 / this.numBatches

        return optimizer
