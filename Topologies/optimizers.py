import warnings

import numpy as np
import numexpr as ne

from typing import *

from NeuralNetworks import *


class WBOptimizer:
    @staticmethod
    def gradientDecent(this, learningRate=0.01):
        learningRate = np.float32(learningRate)

        def optimizer(layer):
            this.deltaBiases[layer] *= learningRate
            this.deltaWeights[layer] *= learningRate

        return optimizer

    @staticmethod
    def momentum(this: "ArtificialNeuralNetwork", learningRate=0.01, alpha=0.001):
        learningRate = np.float32(learningRate)
        ALPHA = np.float32(alpha)
        this.pdb = [0 for _ in range(this.wbShape.LAYERS)]  # pdb -> prev_delta_biases
        this.pdw = this.pdb.copy()  # pdw -> prev_delta_weights  # noqa

        def optimizer(layer):
            this.deltaBiases[layer] = this.pdb[layer] = ALPHA * this.pdb[layer] + learningRate * this.deltaBiases[layer]  # noqa
            this.deltaWeights[layer] = this.pdw[layer] = ALPHA * this.pdw[layer] + learningRate * this.deltaWeights[layer]  # noqa

        return optimizer

    # non verified algorithm
    @staticmethod
    def nesterovMomentum(this: "ArtificialNeuralNetwork", learningRate=0.01, alpha=0.001):
        warnings.showwarning("\nnesterovMomentum has tag 'non verified algorithm' and might not work as intended, "
                             "\nuse 'momentum' instead for stable working", FutureWarning,
                             'optimizers.py->Optimizer.nesterovMomentum', 0)
        learningRate = np.float32(learningRate)
        ALPHA = np.float32(alpha)
        this.pdb = [0 for _ in range(this.wbShape.LAYERS)]  # pdb -> prev_delta_biases
        this.pdw = this.pdb.copy()  # pdw -> prev_delta_weights # noqa
        this.mmb = copyNumpyList(this.biasesList)  # mmb -> momentum_biases
        this.mmw = copyNumpyList(this.weightsList)  # mmw -> momentum_weights # noqa

        def _evalDelta(self, layer):
            deltaBiases = self.deltaLoss[layer] * self.wbActivationDerivatives[layer](self.wbOutputs[layer])
            np.einsum('lkj,lij->ik', self.wbOutputs[layer - 1], deltaBiases, out=self.deltaWeights[layer])
            np.einsum('lij->ij', deltaBiases, out=self.deltaBiases[layer])
            self.deltaLoss[layer - 1] = self.mmw[layer].transpose() @ self.deltaLoss[layer]

        def _fire(self, layer):
            if self.training:
                self.wbOutputs[layer] = self.wbActivations[layer](this.mmw[layer] @ self.wbOutputs[layer - 1] +  # noqa
                                                                  this.mmb[layer])  # noqa
            else:
                super(ArtificialNeuralNetwork, self)._fire(layer)  # noqa

        this._evalDelta = _evalDelta.__get__(this, this.__class__)
        this._fire = _fire.__get__(this, this.__class__)

        def optimizer(layer):
            this.deltaBiases[layer] = this.pdb[layer] = ALPHA * this.pdb[layer] + learningRate * this.deltaBiases[layer]  # noqa
            this.deltaWeights[layer] = this.pdw[layer] = ALPHA * this.pdw[layer] + learningRate * this.deltaWeights[layer]  # noqa

            this.mmb[layer] = this.biasesList[layer] - ALPHA * this.pdb[layer]  # noqa
            this.mmw[layer] = this.weightsList[layer] - ALPHA * this.pdw[layer]  # noqa

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

    @staticmethod
    def adagrad(this: 'ArtificialNeuralNetwork', learning_rate=0.01, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        EPSILON = np.float32(epsilon)
        this.initialize = True
        this.gsqB = [0 for _ in range(this.wbShape.LAYERS)]  # gsqB -> grad_square_biases
        this.gsqW = this.gsqB.copy()  # gsqW -> grad_square_weights # noqa

        def optimizer(layer):
            local_dict = {'deltaBiases': this.deltaBiases[layer], 'gsqB': this.gsqB[layer],  # noqa
                          'deltaWeights': this.deltaWeights[layer], 'gsqW': this.gsqW[layer],  # noqa
                          'EPSILON': EPSILON, 'LEARNING_RATE': LEARNING_RATE}
            this.gsqB[layer] = ne.evaluate('gsqB + deltaBiases*deltaBiases', local_dict=local_dict)  # noqa
            this.gsqW[layer] = ne.evaluate('gsqW + deltaWeights*deltaWeights', local_dict=local_dict)  # noqa

            this.deltaBiases[layer] = ne.evaluate('deltaBiases * LEARNING_RATE / sqrt(gsqB + EPSILON)',
                                                  local_dict=local_dict)
            this.deltaWeights[layer] = ne.evaluate('deltaWeights * LEARNING_RATE / sqrt(gsqW + EPSILON)',
                                                   local_dict=local_dict)

        return optimizer

    @staticmethod
    def rmsprop(this: 'ArtificialNeuralNetwork', learning_rate=0.001, beta=0.9, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        EPSILON = np.float32(epsilon)
        BETA = np.float32(beta)
        BETA_BAR = np.float32(1 - beta)
        this.gsqB = [0 for _ in range(this.wbShape.LAYERS)]  # gsqB -> grad_square_biases
        this.gsqW = this.gsqB.copy()  # gsqW -> grad_square_weights # noqa

        def optimizer(layer):
            local_dict = {'deltaBiases': this.deltaBiases[layer], 'gsqB': this.gsqB[layer],  # noqa
                          'deltaWeights': this.deltaWeights[layer], 'gsqW': this.gsqW[layer],  # noqa
                          'EPSILON': EPSILON, 'LEARNING_RATE': LEARNING_RATE}
            this.gsqB[layer] = BETA * this.gsqB[layer] + \
                               BETA_BAR * np.einsum('ij,ij->ij', this.deltaBiases[layer], this.deltaBiases[layer])
            this.gsqW[layer] = BETA * this.gsqW[layer] + \
                               BETA_BAR * np.einsum('ij,ij->ij', this.deltaWeights[layer], this.deltaWeights[layer])

            this.deltaBiases[layer] = ne.evaluate('deltaBiases * LEARNING_RATE / sqrt(gsqB + EPSILON)',
                                                  local_dict=local_dict)
            this.deltaWeights[layer] = ne.evaluate('deltaWeights * LEARNING_RATE / sqrt(gsqW + EPSILON)',
                                                   local_dict=local_dict)

        return optimizer
