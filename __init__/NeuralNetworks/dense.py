from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools import *
    from ..Topologies import *

import numpy as np

from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape
from ..tools import Network


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


class DenseLayer(BaseLayer):
    """

    """
    def __save__(self):
        return super(DenseLayer, self).__save__()

    def _initializeDepOptimizer(self):
        self.weightOptimizer = self.OPTIMIZER.__new_copy__()
        self.biasesOptimizer = self.OPTIMIZER.__new_copy__()

    def _defineDeps(self) -> list['str']:
        self.weights = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], self.SHAPE.INPUT[0]),
                                                       self.SHAPE.OUTPUT))
        self.biases = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], 1), self.SHAPE.OUTPUT))
        self.deltaWeights = None
        self.deltaBiases = None
        self._initializeDepOptimizer()
        return ['weights', 'biases']

    def _fireAndFindOutput(self) -> "np.ndarray":
        return self.weights @ self.input + self.biases

    def _wireAndFindDelta(self) -> "np.ndarray":
        self.deltaWeights = np.einsum('lij,loj->oi', self.input, self.givenDelta, optimize='greedy')
        self.deltaBiases = self.givenDelta.sum(axis=0)
        delta = self.weights.transpose() @ self.givenDelta

        self.weights -= self.weightOptimizer(self.deltaWeights)
        self.biases -= self.biasesOptimizer(self.deltaBiases)

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
                 lossFunction: "LossFunction.Abstract" = None):
        super(DenseNN, self).__init__(shape, initializers, activators, lossFunction)

    def _constructNetwork(self) -> "Network":
        pass
