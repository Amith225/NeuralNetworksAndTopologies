import cProfile as cP
import time as tm
from abc import ABCMeta, abstractmethod
from typing import *

import numpy as np

from .printVars import PrintVars as pV

if TYPE_CHECKING:
    from . import *
    from Topologies import *


class AbstractNeuralNetwork(metaclass=ABCMeta):
    SMOOTH_PRINT_INTERVAL = 0.25

    def __init__(self):
        self.costs = []
        self.timeTrained = 0
        self.epochs = 1
        self.batchSize = 32
        self.numBatches = None
        self.trainDatabase = None
        self.lossFunction = None

    @abstractmethod
    def _forwardPass(self, layer=1):
        pass

    @abstractmethod
    def _backPropagation(self, layer=-1):
        pass

    @abstractmethod
    def process(self, inputs):
        pass

    @abstractmethod
    def _fire(self, layer):
        pass

    @abstractmethod
    def _wire(self, layer):
        pass

    @abstractmethod
    def _trainer(self, batch) -> "float":
        pass

    def _resetVars(self):
        pass

    def train(self, profile=False):
        if not profile:
            if len(self.costs) == 0:
                costs = [float('nan')]
            else:
                costs = [self.costs[-1][-1]]
            totTime = 0
            waitTime = self.SMOOTH_PRINT_INTERVAL
            self.numBatches = int(np.ceil(self.trainDatabase.size / self.batchSize))
            lastEpoch = self.epochs - 1
            for epoch in range(self.epochs):
                cost = 0
                time = tm.time()
                batchGenerator = self.trainDatabase.batchGenerator(self.batchSize)
                for batch in range(self.numBatches):
                    cost += self._trainer(batchGenerator.__next__())
                time = tm.time() - time
                totTime += time
                cost /= self.trainDatabase.size
                costs.append(cost)
                if totTime > waitTime or epoch == lastEpoch:
                    waitTime += self.SMOOTH_PRINT_INTERVAL
                    print(end='\r')
                    print(pV.CBOLD + pV.CBLUE + pV.CURL + f'epoch:{epoch}' + pV.CEND,
                          pV.CYELLOW + f'cost:{cost}', f'time:{time}' + pV.CEND,
                          pV.CBOLD + pV.CITALIC + pV.CBEIGE + f'cost_reduction:{(costs[-2] - cost)}' + pV.CEND,
                          pV.CBOLD + f'eta:{totTime / (epoch + 1) * (self.epochs - epoch - 1)}',
                          pV.CEND, end='')
            print('\n' + pV.CBOLD + pV.CRED2 + f'totTime:{totTime}', f'avg_time:{totTime / self.epochs}' + pV.CEND)
            self.timeTrained += totTime
            self.costs.append(costs[1:])
        else:
            cP.runctx("self.train()", globals=globals(), locals=locals())

        self._resetVars()


class ArtificialNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, wbShape: "WBShape",
                 wbInitializer: "WBInitializer",
                 wbActivations: "Activations"):
        super(ArtificialNeuralNetwork, self).__init__()
        self.wbShape = wbShape
        self.wbInitializer = wbInitializer
        self.wbActivations, self.wbActivationDerivatives = wbActivations.get(self.wbShape.LAYERS - 1)

        self.biasesList, self.weightsList = self.wbInitializer.initialize(self.wbShape)

        self.optimizer = None

        self.wbOutputs, self.target = list(range(self.wbShape.LAYERS)), None
        self.deltaBiases, self.deltaWeights, self.deltaLoss = None, None, None

    def _resetVars(self):
        self.wbOutputs, self.target = list(range(self.wbShape.LAYERS)), None
        self.deltaBiases, self.deltaWeights, self.deltaLoss = None, None, None

    def _forwardPass(self, layer=1):
        self._fire(layer)
        if layer < self.wbShape.LAYERS - 1:
            self._forwardPass(layer + 1)

    def _backPropagation(self, layer=-1):
        if layer <= -self.wbShape.LAYERS:
            return
        deltaBiases = (self.deltaLoss[layer] * self.wbActivationDerivatives[layer].__call__(self.wbOutputs[layer]))
        np.einsum('lkj,lij->ik', self.wbOutputs[layer - 1], deltaBiases, out=self.deltaWeights[layer])
        np.einsum('lij->ij', deltaBiases, out=self.deltaBiases[layer])
        self.deltaLoss[layer - 1] = self.weightsList[layer].transpose() @ self.deltaLoss[layer]
        self.optimizer(layer)
        self._wire(layer)
        self._backPropagation(layer - 1)

    def process(self, inputs):
        self.wbOutputs[0] = inputs
        self._forwardPass()

        return self.wbOutputs[-1]

    def _fire(self, layer):
        self.wbOutputs[layer] = self.wbActivations[layer - 1](self.weightsList[layer] @ self.wbOutputs[layer - 1] +
                                                              self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        self.weightsList[layer] -= self.deltaWeights[layer]

    def __deltaInitializer(self):
        deltaLoss = [(np.zeros((self.batchSize, self.wbShape[i], 1), dtype=np.float32))
                     for i in range(0, self.wbShape.LAYERS)]
        deltaBiases, deltaWeights = self.wbInitializer.initialize(self.wbShape)

        return deltaBiases, deltaWeights, deltaLoss

    def _trainer(self, batch):
        self.wbOutputs[0], self.target = batch
        self._forwardPass()
        loss, self.deltaLoss[-1] = self.lossFunction.eval(self.wbOutputs[-1], self.target)
        self._backPropagation()

        return loss

    def train(self, epochs: "int" = None, batchSize: "int" = None,
              trainDatabase: "DataBase" = None, lossFunction: "LossFunction" = None,
              optimizer: "Optimizer" = None,
              profile: "bool" = False):
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
        if trainDatabase is not None:
            self.trainDatabase = trainDatabase
        if lossFunction is not None:
            self.lossFunction = lossFunction
        if optimizer is not None:
            self.optimizer = optimizer

        self.deltaBiases, self.deltaWeights, self.deltaLoss = self.__deltaInitializer()
        super(ArtificialNeuralNetwork, self).train(profile)
