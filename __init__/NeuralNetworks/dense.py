from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Topologies import *

import numpy as np

from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape, Network


class DenseShape(BaseShape):
    """

    """

    def __init__(self, *shapes: int):
        super(DenseShape, self).__init__(*shapes)

    @staticmethod
    def _formatShapes(shapes) -> tuple:
        assert len(shapes) > 0
        formattedShape = []
        for s in shapes:
            assert isinstance(s, int) and s > 0, "all args of *shapes must be integers > 0"
            formattedShape.append((s, 1))
        return tuple(formattedShape)


class DenseLayer(BaseLayer):  # todo: pre-set deltas after forwardPass
    """

    """

    def __save__(self):
        return super(DenseLayer, self).__save__()

    def _initializeDepOptimizer(self):
        self.weightOptimizer = self.optimizer.__new_copy__()
        self.biasesOptimizer = self.optimizer.__new_copy__()

    def _defineDeps(self) -> list['str']:
        self.weights = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], self.SHAPE.INPUT[0]),
                                                       self.SHAPE.OUTPUT))
        self.biases = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], 1), self.SHAPE.OUTPUT))
        self._initializeDepOptimizer()
        return ['weights', 'biases']

    def _fire(self) -> "np.ndarray":
        return self.ACTIVATION_FUNCTION.activation(self.weights @ self.input + self.biases)

    def _wireAndFindDelta(self) -> "np.ndarray":
        delta = self.weights.transpose() @ self.inputDelta

        activeDerivedDelta = self.inputDelta * self.ACTIVATION_FUNCTION.activatedDerivative(self.output)
        deltaWeights = np.einsum('lij,loj->oi', self.input, activeDerivedDelta, optimize='greedy')
        deltaBiases = activeDerivedDelta.sum(axis=0)

        self.weights -= self.weightOptimizer(deltaWeights)
        self.biases -= self.biasesOptimizer(deltaBiases)

        return delta


class DensePlot(BasePlot):
    """

    """


class DenseNN(BaseNN):
    """

    """

    def __str__(self):
        return super(DenseNN, self).__str__()

    def __save__(self):
        return super(DenseNN, self).__save__()

    def __init__(self, shape: "DenseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Base" = None):
        super(DenseNN, self).__init__(shape, initializers, activators, lossFunction)

    def _constructNetwork(self) -> "Network":
        layers = []
        for i, _initializer, _optimizer, _aF in zip(range(_length := self.SHAPE.LAYERS - 1),
                                                    self.INITIALIZERS(_length),
                                                    self.optimizers(_length),  # noqa
                                                    self.ACTIVATORS(_length)):
            layers.append(DenseLayer(self.SHAPE[i:i + 2], _initializer, _optimizer, _aF))
        return Network(*layers, lossFunction=self.LOSS_FUNCTION)
