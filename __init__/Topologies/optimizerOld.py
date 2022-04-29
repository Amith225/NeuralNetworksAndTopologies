import warnings as wr
from typing import *
if TYPE_CHECKING:
    from ..NeuralNetworks import DenseNN

import numpy as np
import numexpr as ne


class RmspropWBOptimizer:
    def __init__(self, neural_network: 'DenseNN', learningRate=0.001, beta=0.9, epsilon=np.e ** -8):
        super(RmspropWBOptimizer, self).__init__(neural_network, learningRate, beta=beta, epsilon=epsilon)
        self.grad_square_biases_sum = [0 for _ in range(self.nn.shape.NUM_LAYERS)]
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
class AdadeltaWBOptimizer:
    def __init__(self, neural_network: 'DenseNN', learningRate=0.0001, beta=0.9, epsilon=np.e ** -8):
        wr.showwarning("\nAdadelta has tag 'non verified algorithm' and might not work as intended, "
                       "\nuse 'Rmsprop' instead for stable working", PendingDeprecationWarning,
                       'optimizerOld.py->AdadeltaWBOptimizer', 0)
        super(AdadeltaWBOptimizer, self).__init__(neural_network, learningRate, beta=beta, epsilon=epsilon)
        self.grad_square_biases_sum = [0 for _ in range(self.nn.shape.NUM_LAYERS)]
        self.grad_square_weights_sum = self.grad_square_biases_sum.copy()
        self.delta_square_biases_sum = [0 for _ in range(self.nn.shape.NUM_LAYERS)]
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
