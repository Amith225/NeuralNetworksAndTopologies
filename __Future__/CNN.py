import numpy as np
from ..NeuralNetworks import AbstractNeuralNetwork
from ..Topologies import _
from ..Utils import _


class ConvolutionalNeuralNetwork(AbstractNeuralNetwork):
    def _forwardPass(self, layer=1):
        pass

    def _backPropagate(self, layer=-1):
        pass

    def _fire(self, layer):
        pass

    def _wire(self, layer):
        pass

    def _initializeVars(self):
        pass

    def _trainer(self):
        """assign self.loss and self.deltaLoss here"""

    def __init__(self, shape, activators, costFunction):
        super(ConvolutionalNeuralNetwork, self).__init__(shape, activators, costFunction)
