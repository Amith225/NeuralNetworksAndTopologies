import cProfile as _cP
import time as _tm
import warnings as _wr
import typing as _tp
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np

from .printVars import PrintVars as _pV

if _tp.TYPE_CHECKING:
    from utils import WBShape, Activators
    from Topologies import WBInitializer, WBOptimizer, LossFunction, DataBase


class AbstractNeuralNetwork(metaclass=_ABCMeta):
    SMOOTH_PRINT_INTERVAL = 0.25

    def __init__(self):
        self.costHistory = []
        self.timeTrained = 0
        self.epochs = 1
        self.batchSize = 32
        self.trainAccuracy = float('nan')
        self.testAccuracy = float('nan')
        self.numBatches = None
        self.trainDatabase = None
        self.lossFunction = None
        self.training = False
        self.profiling = False

    @_abstractmethod
    def _forwardPass(self, layer=1):
        pass

    @_abstractmethod
    def _backPropagate(self, layer=-1):
        pass

    @_abstractmethod
    def process(self, inputs) -> "_np.ndarray":
        if self.training:
            _wr.showwarning("can't process while training in progress", ResourceWarning,
                            'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return _np.NAN

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

    def train(self, profile=False, test=None):
        if not profile:
            self.training = True
            if len(self.costHistory) == 0:
                costs = [float('nan')]
            else:
                costs = [self.costHistory[-1][-1]]
            totTime = 0
            waitTime = self.SMOOTH_PRINT_INTERVAL
            self.numBatches = int(_np.ceil(self.trainDatabase.size / self.batchSize))
            lastEpoch = self.epochs
            for epoch in range(1, self.epochs + 1):
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
                    avgTime = totTime / epoch
                    print(end='\r')
                    print(_pV.CBOLDITALICURL + _pV.CBLUE + f'Epoch:{epoch}/{self.epochs}' + _pV.CEND,
                          _pV.CYELLOW + f'Cost:{round(cost, 8):.8f}',
                          f'Cost-Reduction:{round(costs[-2] - cost, 8):.8f}' + _pV.CEND,
                          _pV.CBOLD + _pV.CRED2 + f"Time:{self.secToHMS(time)}",
                          f"Average-Time:{self.secToHMS(avgTime)}",
                          f"Eta:{self.secToHMS(avgTime * (self.epochs - epoch))}",
                          f"Elapsed:{self.secToHMS(totTime)}" + _pV.CEND, end='')
            print()
            self.timeTrained += totTime
            self.costHistory.append(costs[1:])
            self.training = False
            self._resetVars()
        else:
            self.profiling = True
            _cP.runctx("self.train()", globals=globals(), locals=locals())
            self.profiling = False

        if not self.profiling:
            self.test(test)

    @staticmethod
    def secToHMS(seconds):
        encode = '%Ssec'
        if (tim := _tm.gmtime(seconds)).tm_min != 0:
            encode = '%Mmin' + encode
        if tim.tm_hour != 0:
            encode = '%Hhr' + encode

        return _tm.strftime(encode, tim)

    def _accuracy(self, inputSet, targetSet, tarShape, size):
        try:
            out = self.process(inputSet)
            tar = targetSet
            if tarShape != 1:
                outIndex = _np.where(out == _np.max(out, axis=1, keepdims=True))[1]
                targetIndex = _np.where(tar == 1)[1]
            else:
                outIndex = _np.round(out)
                targetIndex = tar
            result = outIndex == targetIndex
            result = _np.float32(1) * result
        except MemoryError:
            accuracy1 = self._accuracy(inputSet[:(to := size // 2)], targetSet[:to], tarShape, to)
            accuracy2 = self._accuracy(inputSet[to:], targetSet[to:], tarShape, size - to)

            return (accuracy1 + accuracy2) / 2

        return result.sum() / result.shape[0] * 100

    def accuracy(self, db: "DataBase"):
        return self._accuracy(db.inputSet[:], db.targetSet[:], db.tarShape, db.size)

    def test(self, testDataBase: "DataBase" = None):
        if self.trainDatabase is not None:
            self.trainAccuracy = self.accuracy(self.trainDatabase)
        if testDataBase is not None:
            self.testAccuracy = self.accuracy(testDataBase)

        print(_pV.CBOLD + _pV.CYELLOW + f'Train-Accuracy:{self.trainAccuracy}%', '\n'
              f'Test-Accuracy :{self.testAccuracy}%' + _pV.CEND)


class ArtificialNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, wbShape: "WBShape",
                 wbInitializer: "WBInitializer",
                 activators: "Activators"):
        super(ArtificialNeuralNetwork, self).__init__()
        self.wbShape = wbShape
        self.wbInitializer = wbInitializer
        self.activations, self.activationDerivatives = activators.get(self.wbShape.LAYERS - 1)

        self.biasesList, self.weightsList = self.wbInitializer(self.wbShape)

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
        self.wbOptimizer(layer)
        self._wire(layer)
        self._backPropagate(layer - 1)

    def process(self, inputs):
        super(ArtificialNeuralNetwork, self).process(inputs)
        inputs = _np.array(inputs)
        if inputs.size % self.wbShape[0] == 0:
            inputs = inputs.reshape([-1, self.wbShape[0], 1])
        else:
            raise Exception("InputError: size of input should be same as that of input node of neural network or "
                            "an integral multiple of it")
        self.wbOutputs[0] = inputs
        self._forwardPass()

        return self.wbOutputs[-1]

    def _fire(self, layer):
        self.wbOutputs[layer] = self.activations[layer](self.weightsList[layer] @ self.wbOutputs[layer - 1] +
                                                        self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        self.weightsList[layer] -= self.deltaWeights[layer]

    def __deltaInitializer(self):
        deltaLoss = [(_np.zeros((self.batchSize, self.wbShape[i], 1), dtype=_np.float32))
                     for i in range(0, self.wbShape.LAYERS)]
        deltaBiases, deltaWeights = self.wbInitializer(self.wbShape)

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
              profile: "bool" = False,
              test: "DataBase" = None):
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
        super(ArtificialNeuralNetwork, self).train(profile, test)
