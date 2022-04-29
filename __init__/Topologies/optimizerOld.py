import warnings as wr
from typing import *
if TYPE_CHECKING:
    from ..NeuralNetworks import DenseNN
from abc import ABCMeta as ABCMeta, abstractmethod as abstractmethod

import numpy as np
import numexpr as ne

from ..tools import copyNumpyList


class WBOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, neural_network: "DenseNN", learningRate, alpha=float('nan'), beta=float('nan'),
                 epsilon=float('nan'), *args, **kwargs):
        self.nn = neural_network
        self.LEARNING_RATE = np.float32(learningRate)
        self.EPSILON = np.float32(epsilon)
        self.ALPHA = np.float32(alpha)
        self.BETA = np.float32(beta)
        self.BETA_BAR = np.float32(1 - beta)

    def __call__(self, layer):
        return self._optimize(layer)

    @abstractmethod
    def _optimize(self, layer):
        self._evalDelta(layer)

    # fixme: call this in a better manner
    def _evalDelta(self, layer):
        deltaBiases = self.nn.deltaLoss[layer] * self.nn.activationDerivatives[layer](self.nn.outputs[layer])
        np.einsum('lkj,lij->ik', self.nn.outputs[layer - 1], deltaBiases, out=self.nn.deltaWeights[layer])
        self.nn.deltaBiases[layer] = deltaBiases.sum(axis=0)
        self.nn.deltaLoss[layer - 1] = self.nn.weightsList[layer].transpose() @ self.nn.deltaLoss[layer]


# non verified algorithm
class NesterovMomentumWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: "DenseNN", learningRate=0.001, alpha=0.9):
        wr.showwarning("\nNesterovMomentum has tag 'non verified algorithm' and might not work as intended, "
                       "\nuse 'momentum' instead for stable working", PendingDeprecationWarning,
                       'optimizerOld.py->NesterovMomentumWBOptimizer', 0)
        if alpha is None:
            alpha = learningRate / 10
        super(NesterovMomentumWBOptimizer, self).__init__(neural_network, learningRate, alpha)
        self.prev_delta_biases = [0 for _ in range(self.nn.shape.LAYERS)]
        self.prev_delta_weights = self.prev_delta_biases.copy()
        self.momentum_biases = copyNumpyList(self.nn.biasesList)
        self.momentum_weights = copyNumpyList(self.nn.weightsList)

        self.nn._fire = self._fire

    def _fire(self, layer):
        if self.nn.training:
            self.nn.outputs[layer] = \
                self.nn.activations[layer](self.momentum_weights[layer] @ self.nn.outputs[layer - 1] +
                                           self.momentum_biases[layer])
        else:
            super(self.nn.__class__, self.nn)._fire(layer)  # noqa

    def _evalDelta(self, layer):
        deltaBiases = self.nn.deltaLoss[layer] * self.nn.activationDerivatives[layer](self.nn.outputs[layer])
        np.einsum('lkj,lij->ik', self.nn.outputs[layer - 1], deltaBiases, out=self.nn.deltaWeights[layer])
        np.einsum('lij->ij', deltaBiases, out=self.nn.deltaBiases[layer])
        self.nn.deltaLoss[layer - 1] = self.momentum_weights[layer].transpose() @ self.nn.deltaLoss[layer]

    def _optimize(self, layer):
        super(NesterovMomentumWBOptimizer, self)._optimize(layer)
        self.nn.deltaBiases[layer] = self.prev_delta_biases[layer] = \
            self.ALPHA * self.prev_delta_biases[layer] + self.LEARNING_RATE * self.nn.deltaBiases[layer]
        self.nn.deltaWeights[layer] = self.prev_delta_weights[layer] = \
            self.ALPHA * self.prev_delta_weights[layer] + self.LEARNING_RATE * self.nn.deltaWeights[layer]

        self.momentum_biases[layer] = self.nn.biasesList[layer] - self.ALPHA * self.prev_delta_biases[layer]
        self.momentum_weights[layer] = self.nn.weightsList[layer] - self.ALPHA * self.prev_delta_weights[layer]


class RmspropWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: 'DenseNN', learningRate=0.001, beta=0.9, epsilon=np.e ** -8):
        super(RmspropWBOptimizer, self).__init__(neural_network, learningRate, beta=beta, epsilon=epsilon)
        self.grad_square_biases_sum = [0 for _ in range(self.nn.shape.LAYERS)]
        self.grad_square_weights_sum = self.grad_square_biases_sum.copy()

    def _optimize(self, layer):
        super(RmspropWBOptimizer, self)._optimize(layer)
        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias_sum': self.grad_square_biases_sum[layer],
                      'grad_square_weight_sum': self.grad_square_weights_sum[layer],
                      'BETA': self.BETA,
                      'BETA_BAR': self.BETA_BAR}
        self.grad_square_biases_sum[layer] = ne.evaluate(
            "BETA * grad_square_bias_sum + BETA_BAR * deltaBias*deltaBias", local_dict=local_dict)
        self.grad_square_weights_sum[layer] = ne.evaluate(
            "BETA * grad_square_weight_sum + BETA_BAR * deltaWeight*deltaWeight", local_dict=local_dict)

        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias_sum': self.grad_square_biases_sum[layer],
                      'grad_square_weight_sum': self.grad_square_weights_sum[layer],
                      'EPSILON': self.EPSILON,
                      'LEARNING_RATE': self.LEARNING_RATE}
        self.nn.deltaBiases[layer] = ne.evaluate('deltaBias * LEARNING_RATE / sqrt(grad_square_bias_sum + EPSILON)',
                                                 local_dict=local_dict)
        self.nn.deltaWeights[layer] = ne.evaluate(
            'deltaWeight * LEARNING_RATE / sqrt(grad_square_weight_sum +EPSILON)', local_dict=local_dict)


# non verified algorithm
class AdadeltaWBOptimizer(WBOptimizer):
    def __init__(self, neural_network: 'DenseNN', learningRate=0.0001, beta=0.9, epsilon=np.e ** -8):
        wr.showwarning("\nAdadelta has tag 'non verified algorithm' and might not work as intended, "
                       "\nuse 'Rmsprop' instead for stable working", PendingDeprecationWarning,
                       'optimizerOld.py->AdadeltaWBOptimizer', 0)
        super(AdadeltaWBOptimizer, self).__init__(neural_network, learningRate, beta=beta, epsilon=epsilon)
        self.grad_square_biases_sum = [0 for _ in range(self.nn.shape.LAYERS)]
        self.grad_square_weights_sum = self.grad_square_biases_sum.copy()
        self.delta_square_biases_sum = [0 for _ in range(self.nn.shape.LAYERS)]
        self.delta_square_weights_sum = self.delta_square_biases_sum.copy()

    def _optimize(self, layer):
        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'grad_square_bias_sum': self.grad_square_biases_sum[layer],
                      'grad_square_weight_sum': self.grad_square_weights_sum[layer],
                      'BETA': self.BETA,
                      'BETA_BAR': self.BETA_BAR}
        self.grad_square_biases_sum[layer] = ne.evaluate(
            "BETA * grad_square_bias_sum + BETA_BAR * deltaBias*deltaBias", local_dict=local_dict)
        self.grad_square_weights_sum[layer] = ne.evaluate(
            "BETA * grad_square_weight_sum + BETA_BAR * deltaWeight*deltaWeight", local_dict=local_dict)

        self.nn.deltaBiases[layer] *= \
            self.LEARNING_RATE * \
            np.sqrt((self.delta_square_biases_sum[layer] + self.EPSILON) / (self.grad_square_biases_sum[layer] +
                                                                            self.EPSILON))
        self.nn.deltaWeights[layer] *= \
            self.LEARNING_RATE * \
            np.sqrt((self.delta_square_weights_sum[layer] + self.EPSILON) / (self.grad_square_weights_sum[layer] +
                                                                             self.EPSILON))

        local_dict = {'deltaBias': self.nn.deltaBiases[layer],
                      'deltaWeight': self.nn.deltaWeights[layer],
                      'delta_square_bias_sum': self.delta_square_biases_sum[layer],
                      'delta_square_weight_sum': self.delta_square_weights_sum[layer],
                      'BETA': self.BETA,
                      'BETA_BAR': self.BETA_BAR}
        self.delta_square_biases_sum[layer] = ne.evaluate(
            "BETA * delta_square_bias_sum + BETA_BAR * deltaBias*deltaBias", local_dict=local_dict)
        self.delta_square_weights_sum[layer] = ne.evaluate(
            "BETA * delta_square_weight_sum + BETA_BAR * deltaWeight*deltaWeight", local_dict=local_dict)
