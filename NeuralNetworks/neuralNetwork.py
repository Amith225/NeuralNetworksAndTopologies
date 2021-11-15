import cProfile as _cP
import time as _tm
import warnings as _wr
import typing as _tp
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np

from .printVars import PrintVars as _pV

if _tp.TYPE_CHECKING:
    from NeuralNetworks import WBShape, Activations, DataBase
    from Topologies import WBInitializer, WBOptimizer, LossFunction


class AbstractNeuralNetwork(metaclass=_ABCMeta):
    SMOOTH_PRINT_INTERVAL = 0.25

    def __init__(self):
        self.costs = []
        self.timeTrained = 0
        self.epochs = 1
        self.batchSize = 32
        self.numBatches = None
        self.trainDatabase = None
        self.lossFunction = None
        self.training = False

    @_abstractmethod
    def _forwardPass(self, layer=1):
        pass

    @_abstractmethod
    def _backPropagate(self, layer=-1):
        pass

    @_abstractmethod
    def process(self, inputs):
        if self.training:
            _wr.showwarning("can't process while training in progress", ResourceWarning,
                            'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return

    @_abstractmethod
    def _fire(self, layer):
        pass

    @_abstractmethod
    def _wire(self, layer):
        pass

    @_abstractmethod
    def _trainer(self, batch) -> "float":
        pass

    def _resetVars(self):
        pass

    def train(self, profile=False):
        self.training = True
        if not profile:
            if len(self.costs) == 0:
                costs = [float('nan')]
            else:
                costs = [self.costs[-1][-1]]
            totTime = 0
            waitTime = self.SMOOTH_PRINT_INTERVAL
            self.numBatches = int(_np.ceil(self.trainDatabase.size / self.batchSize))
            lastEpoch = self.epochs - 1
            for epoch in range(self.epochs):
                cost = 0
                time = _tm.time()
                batchGenerator = self.trainDatabase.batchGenerator(self.batchSize)
                for batch in range(self.numBatches):
                    cost += self._trainer(batchGenerator.__next__())
                time = _tm.time() - time
                totTime += time
                cost /= self.trainDatabase.size
                costs.append(cost)
                if totTime > waitTime or epoch == lastEpoch:
                    waitTime += self.SMOOTH_PRINT_INTERVAL
                    print(end='\r')
                    print(_pV.CBOLD + _pV.CBLUE + _pV.CURL + f'epoch:{epoch}' + _pV.CEND,
                          _pV.CYELLOW + f'cost:{cost}', f'time:{time}' + _pV.CEND,
                          _pV.CBOLD + _pV.CITALIC + _pV.CBEIGE + f'cost-reduction:{(costs[-2] - cost)}' + _pV.CEND,
                          _pV.CBOLD + f'eta:{totTime / (epoch + 1) * (self.epochs - epoch - 1)}', f'elapsed:{totTime}' +
                          _pV.CEND, end='')
            print('\n' + _pV.CBOLD + _pV.CRED2 + f'total-time:{totTime}', f'average-time:{totTime / self.epochs}' +
                  _pV.CEND)
            self.timeTrained += totTime
            self.costs.append(costs[1:])
        else:
            _cP.runctx("self.train()", globals=globals(), locals=locals())

        self._resetVars()
        self.training = False


class ArtificialNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, wbShape: "WBShape",
                 wbInitializer: "WBInitializer",
                 wbActivations: "Activations"):
        super(ArtificialNeuralNetwork, self).__init__()
        self.wbShape = wbShape
        self.wbInitializer = wbInitializer
        self.wbActivations, self.wbActivationDerivatives = wbActivations.get(self.wbShape.LAYERS - 1)

        self.biasesList, self.weightsList = self.wbInitializer.initialize(self.wbShape)

        self.wbOptimizer = None

        self.wbOutputs, self.target = list(range(self.wbShape.LAYERS)), None
        self.deltaBiases, self.deltaWeights, self.deltaLoss = None, None, None

    def _resetVars(self):
        self.wbOutputs, self.target = list(range(self.wbShape.LAYERS)), None
        self.deltaBiases, self.deltaWeights, self.deltaLoss = None, None, None

    def _forwardPass(self, layer=1):
        self._fire(layer)
        if layer < self.wbShape.LAYERS - 1:
            self._forwardPass(layer + 1)

    def _backPropagate(self, layer=-1):
        if layer <= -self.wbShape.LAYERS:
            return
        self.wbOptimizer.optimize(layer)
        self._wire(layer)
        self._backPropagate(layer - 1)

    def process(self, inputs):
        super(ArtificialNeuralNetwork, self).process(inputs)
        inputs = _np.array(inputs)
        if inputs.size % self.wbShape[0] == 0:
            inputs = inputs.reshape([-1, self.wbShape[0], 1])
        else:
            raise Exception("InputError: size of input should be same as that of input node of neural network or"
                            "an integral multiple of it")
        self.wbOutputs[0] = inputs
        self._forwardPass()

        return self.wbOutputs[-1]

    def _fire(self, layer):
        self.wbOutputs[layer] = self.wbActivations[layer](self.weightsList[layer] @ self.wbOutputs[layer - 1] +
                                                          self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        self.weightsList[layer] -= self.deltaWeights[layer]

    def __deltaInitializer(self):
        deltaLoss = [(_np.zeros((self.batchSize, self.wbShape[i], 1), dtype=_np.float32))
                     for i in range(0, self.wbShape.LAYERS)]
        deltaBiases, deltaWeights = self.wbInitializer.initialize(self.wbShape)

        return deltaBiases, deltaWeights, deltaLoss

    def _trainer(self, batch):
        self.wbOutputs[0], self.target = batch
        self._forwardPass()
        loss, self.deltaLoss[-1] = self.lossFunction.eval(self.wbOutputs[-1], self.target)
        self._backPropagate()

        return loss

    def train(self, epochs: "int" = None, batchSize: "int" = None,
              trainDatabase: "DataBase" = None, lossFunction: "LossFunction" = None,
              wbOptimizer: "WBOptimizer" = None,
              profile: "bool" = False):
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
        if trainDatabase is not None:
            self.trainDatabase = trainDatabase
        if lossFunction is not None:
            self.lossFunction = lossFunction
        if wbOptimizer is not None:
            self.wbOptimizer = wbOptimizer

        self.deltaBiases, self.deltaWeights, self.deltaLoss = self.__deltaInitializer()
        super(ArtificialNeuralNetwork, self).train(profile)
