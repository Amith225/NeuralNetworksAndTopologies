import warnings as _wr
import typing as _tp
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np
import numexpr as _ne

from NeuralNetworks import copyNumpyList

if _tp.TYPE_CHECKING:
    from NeuralNetworks import ArtificialNeuralNetwork


class WBOptimizer(metaclass=_ABCMeta):
    @_abstractmethod
    def __init__(self, neural_network: "ArtificialNeuralNetwork", learningRate, alpha=float('nan'), beta=float('nan'),
                 epsilon=float('nan'), *args, **kwargs):
        self.nn = neural_network
        self.LEARNING_RATE = _np.float32(learningRate)
        self.EPSILON = _np.float32(epsilon)
        self.ALPHA = _np.float32(alpha)
        self.BETA = _np.float32(beta)
        self.BETA_BAR = _np.float32(1 - beta)

    @_abstractmethod
    def optimize(self, layer):
        self._evalDelta(layer)
    
    def _evalDelta(self, layer):
        deltaBiases = self.nn.deltaLoss[layer] * self.nn.wbActivationDerivatives[layer](self.nn.wbOutputs[layer])
        _np.einsum('lkj,lij->ik', self.nn.wbOutputs[layer - 1], deltaBiases, out=self.nn.deltaWeights[layer])
        _np.einsum('lij->ij', deltaBiases, out=self.nn.deltaBiases[layer])
        self.nn.deltaLoss[layer - 1] = self.nn.weightsList[layer].transpose() @ self.nn.deltaLoss[layer]


class GradientDecentWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: "ArtificialNeuralNetwork", learningRate=0.01):
        super(GradientDecentWBOptimizer, self).__init__(neural_network, learningRate)

    def optimize(self, layer):
        super(GradientDecentWBOptimizer, self).optimize(layer)
        self.nn.deltaBiases[layer] *= self.LEARNING_RATE
        self.nn.deltaWeights[layer] *= self.LEARNING_RATE


class MomentumWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: "ArtificialNeuralNetwork", learningRate=0.01, alpha=None):
        if alpha is None:
            alpha = learningRate / 10
        super(MomentumWBOptimizer, self).__init__(neural_network, learningRate, alpha)
        self.prev_delta_biases = [0 for _ in range(self.nn.wbShape.LAYERS)]
        self.prev_delta_weights = self.prev_delta_biases.copy()

    def optimize(self, layer):
        super(MomentumWBOptimizer, self).optimize(layer)
        self.nn.deltaBiases[layer] = self.prev_delta_biases[layer] =\
            self.ALPHA * self.prev_delta_biases[layer] + self.LEARNING_RATE * self.nn.deltaBiases[layer]
        self.nn.deltaWeights[layer] = self.prev_delta_weights[layer] =\
            self.ALPHA * self.prev_delta_weights[layer] + self.LEARNING_RATE * self.nn.deltaWeights[layer]


# non verified algorithm
class NesterovMomentumWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: "ArtificialNeuralNetwork", learningRate=0.01, alpha=None):
        _wr.showwarning("\nNesterovMomentum has tag 'non verified algorithm' and might not work as intended, "
                        "\nuse 'momentum' instead for stable working", FutureWarning,
                        'optimizer.py->NesterovMomentumWBOptimizer', 0)
        if alpha is None:
            alpha = learningRate / 10
        super(NesterovMomentumWBOptimizer, self).__init__(neural_network, learningRate, alpha)
        self.prev_delta_biases = [0 for _ in range(self.nn.wbShape.LAYERS)]
        self.prev_delta_weights = self.prev_delta_biases.copy()
        self.momentum_biases = copyNumpyList(self.nn.biasesList)
        self.momentum_weights = copyNumpyList(self.nn.weightsList)

        self.nn._fire = self._fire
    
    def _fire(self, layer):
        if self.nn.training:
            self.nn.wbOutputs[layer] =\
                self.nn.wbActivations[layer](self.momentum_weights[layer] @ self.nn.wbOutputs[layer - 1] +
                                             self.momentum_biases[layer])
        else:
            super(ArtificialNeuralNetwork, self.nn)._fire(layer)  # noqa

    def _evalDelta(self, layer):
        deltaBiases = self.nn.deltaLoss[layer] * self.nn.wbActivationDerivatives[layer](self.nn.wbOutputs[layer])
        _np.einsum('lkj,lij->ik', self.nn.wbOutputs[layer - 1], deltaBiases, out=self.nn.deltaWeights[layer])
        _np.einsum('lij->ij', deltaBiases, out=self.nn.deltaBiases[layer])
        self.nn.deltaLoss[layer - 1] = self.momentum_weights[layer].transpose() @ self.nn.deltaLoss[layer]

    def optimize(self, layer):
        super(NesterovMomentumWBOptimizer, self).optimize(layer)
        self.nn.deltaBiases[layer] = self.prev_delta_biases[layer] =\
            self.ALPHA * self.prev_delta_biases[layer] + self.LEARNING_RATE * self.nn.deltaBiases[layer]
        self.nn.deltaWeights[layer] = self.prev_delta_weights[layer] =\
            self.ALPHA * self.prev_delta_weights[layer] + self.LEARNING_RATE * self.nn.deltaWeights[layer]

        self.momentum_biases[layer] = self.nn.biasesList[layer] - self.ALPHA * self.prev_delta_biases[layer]
        self.momentum_weights[layer] = self.nn.weightsList[layer] - self.ALPHA * self.prev_delta_weights[layer]


class DecayWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: "ArtificialNeuralNetwork", learningRate=0.01, alpha=None):
        if alpha is None:
            alpha = 1 / learningRate
        super(DecayWBOptimizer, self).__init__(neural_network, learningRate, alpha)
        self.ALPHA = _np.float32(alpha)
        self.decayCount = 0

    def optimize(self, layer):
        super(DecayWBOptimizer, self).optimize(layer)
        k = self.LEARNING_RATE / (1 + self.decayCount / self.ALPHA)
        self.nn.deltaBiases[layer] *= k
        self.nn.deltaWeights[layer] *= k

        self.decayCount += 1 / self.nn.numBatches


class AdagradWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: 'ArtificialNeuralNetwork', learningRate=0.01, epsilon=_np.e ** -8):
        super(AdagradWBOptimizer, self).__init__(neural_network, learningRate, epsilon=epsilon)
        self.grad_square_biases = [0 for _ in range(self.nn.wbShape.LAYERS)]
        self.grad_square_weights = self.grad_square_biases.copy()

    def optimize(self, layer):
        super(AdagradWBOptimizer, self).optimize(layer)
        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias': self.grad_square_biases[layer],
                      'grad_square_weight': self.grad_square_weights[layer]}
        self.grad_square_biases[layer] = _ne.evaluate('grad_square_bias + deltaBias*deltaBias',
                                                      local_dict=local_dict)
        self.grad_square_weights[layer] = _ne.evaluate('grad_square_weight + deltaWeight*deltaWeight',
                                                       local_dict=local_dict)

        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias': self.grad_square_biases[layer],
                      'grad_square_weight': self.grad_square_weights[layer],
                      'EPSILON': self.EPSILON,
                      'LEARNING_RATE': self.LEARNING_RATE}
        self.nn.deltaBiases[layer] = _ne.evaluate('deltaBias * LEARNING_RATE / sqrt(grad_square_bias + EPSILON)',
                                                  local_dict=local_dict)
        self.nn.deltaWeights[layer] = _ne.evaluate('deltaWeight * LEARNING_RATE / sqrt(grad_square_weight + EPSILON)',
                                                   local_dict=local_dict)


class RmspropWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: 'ArtificialNeuralNetwork', learningRate=0.001, beta=0.9, epsilon=_np.e ** -8):
        super(RmspropWBOptimizer, self).__init__(neural_network, learningRate, beta=beta, epsilon=epsilon)
        self.grad_square_biases_sum = [0 for _ in range(self.nn.wbShape.LAYERS)]
        self.grad_square_weights_sum = self.grad_square_biases_sum.copy()

    def optimize(self, layer):
        super(RmspropWBOptimizer, self).optimize(layer)
        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias_sum': self.grad_square_biases_sum[layer],
                      'grad_square_weight_sum': self.grad_square_weights_sum[layer],
                      'BETA': self.BETA,
                      'BETA_BAR': self.BETA_BAR}
        self.grad_square_biases_sum[layer] = _ne.evaluate(
            "BETA * grad_square_bias_sum + BETA_BAR * deltaBias*deltaBias", local_dict=local_dict)
        self.grad_square_weights_sum[layer] = _ne.evaluate(
            "BETA * grad_square_weight_sum + BETA_BAR * deltaWeight*deltaWeight", local_dict=local_dict)

        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias_sum': self.grad_square_biases_sum[layer],
                      'grad_square_weight_sum': self.grad_square_weights_sum[layer],
                      'EPSILON': self.EPSILON,
                      'LEARNING_RATE': self.LEARNING_RATE}
        self.nn.deltaBiases[layer] = _ne.evaluate('deltaBias * LEARNING_RATE / sqrt(grad_square_bias_sum + EPSILON)',
                                                  local_dict=local_dict)
        self.nn.deltaWeights[layer] = _ne.evaluate(
            'deltaWeight * LEARNING_RATE / sqrt(grad_square_weight_sum +EPSILON)', local_dict=local_dict)


# non verified algorithm
class AdadeltaWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: 'ArtificialNeuralNetwork', learningRate=0.0001, beta=0.9, epsilon=_np.e ** -8):
        _wr.showwarning("\nAdadelta has tag 'non verified algorithm' and might not work as intended, "
                        "\nuse 'Rmsprop' instead for stable working", FutureWarning,
                        'optimizer.py->AdadeltaWBOptimizer', 0)
        super(AdadeltaWBOptimizer, self).__init__(neural_network, learningRate, beta=beta, epsilon=epsilon)
        self.grad_square_biases_sum = [0 for _ in range(self.nn.wbShape.LAYERS)]
        self.grad_square_weights_sum = self.grad_square_biases_sum.copy()
        self.delta_square_biases_sum = [0 for _ in range(self.nn.wbShape.LAYERS)]
        self.delta_square_weights_sum = self.delta_square_biases_sum.copy()

    def optimize(self, layer):
        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias_sum': self.grad_square_biases_sum[layer],
                      'grad_square_weight_sum': self.grad_square_weights_sum[layer],
                      'BETA': self.BETA,
                      'BETA_BAR': self.BETA_BAR}
        self.grad_square_biases_sum[layer] = _ne.evaluate(
            "BETA * grad_square_bias_sum + BETA_BAR * deltaBias*deltaBias", local_dict=local_dict)
        self.grad_square_weights_sum[layer] = _ne.evaluate(
            "BETA * grad_square_weight_sum + BETA_BAR * deltaWeight*deltaWeight", local_dict=local_dict)

        self.nn.deltaBiases[layer] *=\
            self.LEARNING_RATE *\
            _np.sqrt((self.delta_square_biases_sum[layer] + self.EPSILON) / (self.grad_square_biases_sum[layer] +
                                                                             self.EPSILON))
        self.nn.deltaWeights[layer] *=\
            self.LEARNING_RATE *\
            _np.sqrt((self.delta_square_weights_sum[layer] + self.EPSILON) / (self.grad_square_weights_sum[layer] +
                                                                              self.EPSILON))

        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'delta_square_bias_sum': self.delta_square_biases_sum[layer],
                      'delta_square_weight_sum': self.delta_square_weights_sum[layer],
                      'BETA': self.BETA,
                      'BETA_BAR': self.BETA_BAR}
        self.delta_square_biases_sum[layer] = _ne.evaluate(
            "BETA * delta_square_bias_sum + BETA_BAR * deltaBias*deltaBias", local_dict=local_dict)
        self.delta_square_weights_sum[layer] = _ne.evaluate(
            "BETA * delta_square_weight_sum + BETA_BAR * deltaWeight*deltaWeight", local_dict=local_dict)
