import cProfile as _cP
import time as _tm
import warnings as _wr
import typing as _tp
import dill as _dl
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np

from ._printVars import PrintVars as _pV

from utils import AbstractSave

if _tp.TYPE_CHECKING:
    from utils import WBShape, Activators
    from Topologies import WBInitializer, WBOptimizer, LossFunction, DataBase


class AbstractNeuralNetwork(AbstractSave, metaclass=_ABCMeta):
    DEFAULT_DIR = '\\Models\\'
    DEFAULT_NAME = 'nn'
    FILE_TYPE = '.nnt'
    SMOOTH_PRINT_INTERVAL = 0.25

    def saveName(self) -> str:
        return f"{self.costTrained:.2f}c{self.epochTrained}e{self.secToHMS(self.timeTrained)}"

    def _write(self, dumpFile, *args, **kwargs):
        trainDataBase = self.trainDataBase
        self.trainDataBase = None
        _dl.dump(self, dumpFile)
        self.trainDataBase: "DataBase" = trainDataBase

    def __init__(self):
        self.costHistory = []
        self.costTrained = 0
        self.timeTrained = 0
        self.epochTrained = 0
        self.epochs = 1
        self.batchSize = 32
        self.trainAccuracy = float('nan')
        self.testAccuracy = float('nan')
        self.numBatches = None
        self.trainDataBase = None
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
            _wr.showwarning("processing while training in progress, may have unintended conflicts", ResourceWarning,
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

    @staticmethod
    def _statPrinter(key, value, prefix='', suffix=_pV.CEND, end=' '):
        print(prefix + f"{key}:{value}" + suffix, end=end)

    def train(self, *args, profile=False, test=None, **kwargs):
        if not profile:
            if len(self.costHistory) == 0:
                trainCosts = [self.lossFunction(self.process(self.trainDataBase.inputSet[:self.batchSize]),
                                                self.trainDataBase.targetSet[:self.batchSize])[0]]
            else:
                trainCosts = [self.costHistory[-1][-1]]
            self.training = True
            self.numBatches = int(_np.ceil(self.trainDataBase.size / self.batchSize))
            trainTime = 0
            nextPrintTime = self.SMOOTH_PRINT_INTERVAL
            self._statPrinter('Epoch', f"0/{self.epochs}", prefix=_pV.CBOLDITALICURL + _pV.CBLUE)
            for epoch in range(1, self.epochs + 1):
                self.costTrained = 0
                time = _tm.time()
                batchGenerator = self.trainDataBase.batchGenerator(self.batchSize)
                for batch in range(self.numBatches):
                    self.costTrained += self._trainer(batchGenerator.__next__())
                epochTime = _tm.time() - time
                trainTime += epochTime
                self.costTrained /= self.trainDataBase.size
                trainCosts.append(self.costTrained)
                if trainTime >= nextPrintTime or epoch == self.epochs:
                    nextPrintTime += self.SMOOTH_PRINT_INTERVAL
                    avgTime = trainTime / epoch
                    print(end='\r')
                    self._statPrinter('Epoch', f"{epoch}/{self.epochs}", prefix=_pV.CBOLDITALICURL + _pV.CBLUE)
                    self._statPrinter('Cost', f"{self.costTrained:.8f}", prefix=_pV.CYELLOW, suffix='')
                    self._statPrinter('Cost-Reduction', f"{(trainCosts[-2] - self.costTrained):.8f}")
                    self._statPrinter('Time', self.secToHMS(epochTime), prefix=_pV.CBOLD + _pV.CRED2, suffix='')
                    self._statPrinter('Average-Time', self.secToHMS(avgTime), suffix='')
                    self._statPrinter('Eta', self.secToHMS(avgTime * (self.epochs - epoch)), suffix='')
                    self._statPrinter('Elapsed', self.secToHMS(trainTime))
                self.epochTrained += 1
            print()
            self.timeTrained += trainTime
            self.costHistory.append(trainCosts)
            self.training = False
            self._resetVars()
        else:
            self.profiling = True
            _cP.runctx("self.train()", globals=globals(), locals=locals())
            self.profiling = False

        if not self.profiling:
            self.test(test)

    @staticmethod
    def secToHMS(seconds, hms=('h', 'm', 's')):
        encode = f'%S{hms[2]}'
        if (tim := _tm.gmtime(seconds)).tm_min != 0:
            encode = f'%M{hms[1]}' + encode
        if tim.tm_hour != 0:
            encode = f'%H{hms[0]}' + encode

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
        return self._accuracy(db.inputSet, db.targetSet, db.tarShape, db.size)

    def test(self, testDataBase: "DataBase" = None):
        self._statPrinter('Testing', 'wait...', prefix=_pV.CBOLD + _pV.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            self.trainAccuracy = self.accuracy(self.trainDataBase)
        if testDataBase is not None:
            self.testAccuracy = self.accuracy(testDataBase)
        print(end='\r')
        self._statPrinter('Train-Accuracy', f"{self.trainAccuracy}%", suffix='', end='\n')
        self._statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')


class ArtificialNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, wbShape: "WBShape",
                 wbInitializer: "WBInitializer",
                 activators: "Activators"):
        super(ArtificialNeuralNetwork, self).__init__()
        self.wbShape = wbShape
        self.wbInitializer = wbInitializer
        self.activations, self.activationDerivatives = activators(self.wbShape.LAYERS - 1)

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
        loss, self.deltaLoss[-1] = self.lossFunction(self.wbOutputs[-1], self.target)
        self._backPropagate()

        return loss

    def train(self, epochs: "int" = None, batchSize: "int" = None,
              trainDataBase: "DataBase" = None, lossFunction: "LossFunction" = None,
              wbOptimizer: "WBOptimizer" = None,
              profile: "bool" = False,
              test: "DataBase" = None):
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
        if trainDataBase is not None:
            self.trainDataBase = trainDataBase
        if lossFunction is not None:
            self.lossFunction = lossFunction
        if wbOptimizer is not None:
            self.wbOptimizer = wbOptimizer

        self.deltaBiases, self.deltaWeights, self.deltaLoss = self.__deltaInitializer()
        super(ArtificialNeuralNetwork, self).train(profile, test)
