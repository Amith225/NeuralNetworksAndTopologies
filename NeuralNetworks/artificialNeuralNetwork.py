import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..Topologies import LossFunction, Initializer
    from ..Utils import Shape, Activators
import numpy as np

from .neuralNetwork import AbstractNeuralNetwork


class ArtificialNeuralNetwork(AbstractNeuralNetwork):
    def _forwardPass(self, layer=1):
        self._fire(layer)
        if layer < self.shape.LAYERS - 1:
            self._forwardPass(layer + 1)

    def _backPropagate(self, layer=-1):
        if layer <= -self.shape.LAYERS:
            return
        self.optimizer(layer)
        self._wire(layer)
        self._backPropagate(layer - 1)

    def _fire(self, layer):
        self.outputs[layer] = self.activations[layer](self.weightsList[layer] @ self.outputs[layer - 1] +
                                                      self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        self.weightsList[layer] -= self.deltaWeights[layer]

    def _trainer(self):
        self.loss, self.deltaLoss[-1] = self.lossFunction(self.outputs[-1], self.target)

    def _initializeVars(self):
        self.outputs, self.target = list(range(self.shape.LAYERS)), None
        self.deltaBiases, self.deltaWeights = [np.zeros_like(bias) for bias in self.biasesList],\
                                              [np.zeros_like(weight) for weight in self.weightsList]

    def __init__(self, shape: "Shape",
                 initializer: "Initializer",
                 activators: "Activators",
                 costFunction: "LossFunction"):
        super(ArtificialNeuralNetwork, self).__init__(shape, activators, costFunction)

        # todo: make this abstract requirement?
        self.biasesList = initializer(self.shape)
        self.weightsList = initializer([(s[0], self.shape[i - 1][0]) for i, s in enumerate(self.shape)])

        self._initializeVars()
