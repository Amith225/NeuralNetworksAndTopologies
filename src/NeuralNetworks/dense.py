from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Topologies import *

import numpy as np

from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape, Network


class DenseShape(BaseShape):
    """

    """

    def __init__(self, inputShape: int, *shapes: int):
        super(DenseShape, self).__init__(inputShape, *shapes)

    @staticmethod
    def _formatShapes(shapes) -> tuple:
        formattedShape = []
        for s in shapes:
            assert isinstance(s, int) and s > 0, "all args of *shapes must be integers > 0"
            formattedShape.append((s, 1))
        return tuple(formattedShape)


class DenseLayer(BaseLayer):  # todo: pre-set deltas after forwardPass
    """

    """

    def _initializeDepOptimizer(self):
        self.weightOptimizer = self.optimizer.__new_copy__()
        self.biasesOptimizer = self.optimizer.__new_copy__()

    def _defineDeps(self) -> list['str']:
        self.weight = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], self.SHAPE.INPUT[0]),
                                                      self.SHAPE.OUTPUT))
        self.biases = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *self.SHAPE.OUTPUT, self.SHAPE.OUTPUT))
        self.delta = None
        self.activeDerivedDelta = None
        return ['weight', 'biases']

    def __gradWeights(self, weights):  # BottleNeck
        self.delta = np.einsum('oi,...oj->...ij', weights, self.inputDelta, optimize='greedy')
        self.activeDerivedDelta = \
            np.einsum('...ij,...ij->...ij', self.inputDelta, self.ACTIVATION_FUNCTION.activatedDerivative(self.output),
                      optimize='greedy')
        return np.einsum('...ij,...oj->oi', self.input, self.activeDerivedDelta, optimize='greedy')

    def __gradBiases(self, _=None):
        return self.activeDerivedDelta.sum(axis=0)

    def _fire(self) -> "np.ndarray":  # BottleNeck
        return self.ACTIVATION_FUNCTION.activation(
            np.einsum('oi,...ij->...oj', self.weight, self.input, optimize='greedy') + self.biases)

    def _wire(self) -> "np.ndarray":
        self.weight -= self.weightOptimizer(self.__gradWeights, self.weight)
        self.biases -= self.biasesOptimizer(self.__gradBiases, self.biases)
        return self.delta


class DensePlot(BasePlot):
    """

    """


class DenseNN(BaseNN):
    """

    """

    def __init__(self, shape: "DenseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Base" = None):
        super(DenseNN, self).__init__(shape, initializers, activators, lossFunction)

    def _constructNetwork(self, initializers: "Initializers" = None,  # noqa
                          activators: "Activators" = None,
                          lossFunction: "LossFunction.Base" = None) -> "Network":
        layers = []
        for i, _initializer, _optimizer, _aF in zip(range(_length := self.SHAPE.NUM_LAYERS - 1),
                                                    initializers(_length),
                                                    self.optimizers(_length),  # noqa
                                                    activators(_length)):
            layers.append(DenseLayer(self.SHAPE[i:i + 2], _initializer, _optimizer, _aF))
        return Network(*layers, lossFunction=lossFunction)
