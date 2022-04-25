import numpy as np

from .baseNN import AbstractNN


class DenseNN(AbstractNN):
    def _forwardPass(self, layer=1):
        self._fire(layer)
        if layer < self.shape.LAYERS - 1: self._forwardPass(layer + 1)

    def _backPropagate(self, layer=-1):
        if layer <= -self.shape.LAYERS: return
        self.optimizer(layer)
        self._wire(layer)
        self._backPropagate(layer - 1)

    def _evalDelta(self, layer):
        deltaBiases = self.deltaLoss[layer] * self.activationDerivatives[layer](self.outputs[layer])
        np.einsum('lkj,lij->ik', self.outputs[layer - 1], deltaBiases, out=self.deltaWeights[layer])
        self.deltaBiases[layer] = deltaBiases.sum(axis=0)
        self.deltaLoss[layer - 1] = self.weightsList[layer].transpose() @ self.deltaLoss[layer]

    def _fire(self, layer):
        self.outputs[layer] = self.activations[layer](
            self.weightsList[layer] @ self.outputs[layer - 1] + self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        self.weightsList[layer] -= self.deltaWeights[layer]

    def _findOutputShape(self):
        return self.shape

    def _initializeWeightsBiasesDelta(self):
        self.biasesList = self.initializer(self.shape)
        self.weightsList = self.initializer([(s[0], self.shape[i - 1][0]) for i, s in enumerate(self.shape)])
        self.deltaBiases = [np.zeros_like(bias) for bias in self.biasesList]
        self.deltaWeights = [np.zeros_like(weight) for weight in self.weightsList]
