import typing as tp
if tp.TYPE_CHECKING:
    from utils import *
    from . import *
    from ..Topologies import *
import cProfile as cP
import time as tm
import warnings as wr
import dill as dl

import numpy as np

from abc import ABCMeta as ABCMeta, abstractmethod as abstractmethod

from ._printVars import PrintVars as pV
from utils import AbstractSave


# todo: implement auto save. L
class AbstractNeuralNetwork(AbstractSave, metaclass=ABCMeta):
    DEFAULT_DIR = '\\Models\\'
    DEFAULT_NAME = 'nn'
    FILE_TYPE = '.nnt'
    SMOOTH_PRINT_INTERVAL = 0.25

    def saveName(self) -> str:
        return f"{int(self.costTrained * 100)}c.{self.epochTrained}e.{self.secToHMS(self.timeTrained)}"

    def _write(self, dumpFile, *args, **kwargs):
        self._initializeVars()
        trainDataBase = self.trainDataBase
        self.trainDataBase = None
        dl.dump(self, dumpFile)
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

    @abstractmethod
    def _forwardPass(self, layer=1):
        pass

    @abstractmethod
    def _backPropagate(self, layer=-1):
        pass

    @abstractmethod
    def process(self, inputs) -> "np.ndarray":
        if self.training:
            wr.showwarning("processing while training in progress, may have unintended conflicts", ResourceWarning,
                           'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN

    @abstractmethod
    def _fire(self, layer):
        pass

    @abstractmethod
    def _wire(self, layer):
        pass

    @abstractmethod
    def _trainer(self, batch) -> "float":
        pass

    def _initializeVars(self):
        pass

    @staticmethod
    def _statPrinter(key, value, prefix='', suffix=pV.CEND, end=' '):
        print(prefix + f"{key}:{value}" + suffix, end=end)

    def train(self, *args, profile=False, test=None, **kwargs):
        if not profile:
            if len(self.costHistory) == 0:
                trainCosts = [self.lossFunction(self.process(self.trainDataBase.inputSet[:self.batchSize]),
                                                self.trainDataBase.targetSet[:self.batchSize])[0]]
            else:
                trainCosts = [self.costHistory[-1][-1]]
            self.training = True
            self.numBatches = int(np.ceil(self.trainDataBase.size / self.batchSize))
            trainTime = 0
            nextPrintTime = self.SMOOTH_PRINT_INTERVAL
            self._statPrinter('Epoch', f"0/{self.epochs}", prefix=pV.CBOLDITALICURL + pV.CBLUE)
            for epoch in range(1, self.epochs + 1):
                self.costTrained = 0
                time = tm.time()
                batchGenerator = self.trainDataBase.batchGenerator(self.batchSize)
                for batch in range(self.numBatches):
                    self.costTrained += self._trainer(batchGenerator.__next__())
                epochTime = tm.time() - time
                trainTime += epochTime
                self.costTrained /= self.trainDataBase.size
                trainCosts.append(self.costTrained)
                if trainTime >= nextPrintTime or epoch == self.epochs:
                    nextPrintTime += self.SMOOTH_PRINT_INTERVAL
                    avgTime = trainTime / epoch
                    print(end='\r')
                    self._statPrinter('Epoch', f"{epoch}/{self.epochs}", prefix=pV.CBOLDITALICURL + pV.CBLUE)
                    self._statPrinter('Cost', f"{self.costTrained:.8f}", prefix=pV.CYELLOW, suffix='')
                    self._statPrinter('Cost-Reduction', f"{(trainCosts[-2] - self.costTrained):.8f}")
                    self._statPrinter('Time', self.secToHMS(epochTime), prefix=pV.CBOLD + pV.CRED2, suffix='')
                    self._statPrinter('Average-Time', self.secToHMS(avgTime), suffix='')
                    self._statPrinter('Eta', self.secToHMS(avgTime * (self.epochs - epoch)), suffix='')
                    self._statPrinter('Elapsed', self.secToHMS(trainTime))
                self.epochTrained += 1
            print()
            self.timeTrained += trainTime
            self.costHistory.append(trainCosts)
            self.training = False
            self._initializeVars()
        else:
            self.profiling = True
            cP.runctx("self.train()", globals=globals(), locals=locals())
            self.profiling = False

        if not self.profiling:
            self.test(test)

    @staticmethod
    def secToHMS(seconds, hms=('h', 'm', 's')):
        encode = f'%S{hms[2]}'
        if (tim := tm.gmtime(seconds)).tm_min != 0:
            encode = f'%M{hms[1]}' + encode
        if tim.tm_hour != 0:
            encode = f'%H{hms[0]}' + encode

        return tm.strftime(encode, tim)

    @staticmethod
    def _tester(out, tar):
        if np.shape(tar) != 1:
            outIndex = np.where(out == np.max(out, axis=1, keepdims=True))[1]
            targetIndex = np.where(tar == 1)[1]
        else:
            outIndex = np.round(out)
            targetIndex = tar
        result = outIndex == targetIndex
        result = np.float32(1) * result

        return result.sum() / result.shape[0] * 100

    def accuracy(self, inputSet, targetSet):
        assert (size := np.shape(inputSet)[0]) == np.shape(targetSet)[0], \
            "the size of both inputSet and targetSet should be same"
        try:
            return self._tester(self.process(inputSet), targetSet)
        except MemoryError:
            accuracy1 = self.accuracy(inputSet[:(to := size // 2)], targetSet[:to])
            accuracy2 = self.accuracy(inputSet[to:], targetSet[to:])
            return (accuracy1 + accuracy2) / 2

    def test(self, testDataBase: "DataBase" = None):
        self._statPrinter('Testing', 'wait...', prefix=pV.CBOLD + pV.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            db = self.trainDataBase
            self.trainAccuracy = self.accuracy(db.inputSet, db.targetSet)
        if testDataBase is not None:
            db = testDataBase
            self.testAccuracy = self.accuracy(db.inputSet, db.targetSet)
        print(end='\r')
        self._statPrinter('Train-Accuracy', f"{self.trainAccuracy}%", suffix='', end='\n')
        self._statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')
        self._initializeVars()


class ArtificialNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self, wbShape: "WBShape",
                 wbInitializer: "WBInitializer",
                 activators: "Activators"):
        super(ArtificialNeuralNetwork, self).__init__()
        self.wbShape = wbShape
        self.wbInitializer = wbInitializer
        self.activations, self.activationDerivatives = activators(self.wbShape.LAYERS - 1)

        self.biasesList, self.weightsList = self.wbInitializer(self.wbShape)

        self._initializeVars()
        self.wbOptimizer = None
        self.deltaLoss = None

    def _initializeVars(self):
        self.wbOutputs, self.target = list(range(self.wbShape.LAYERS)), None
        self.deltaBiases, self.deltaWeights = [np.zeros_like(bias) for bias in self.biasesList],\
                                              [np.zeros_like(weight) for weight in self.weightsList]

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
        inputs = np.array(inputs)
        if inputs.size % self.wbShape[0] == 0:
            inputs = inputs.reshape([-1, self.wbShape[0], 1])
        else:
            raise Exception("InputError: size of input should be same as that of input node of neural network or "
                            "an integral multiple of it")
        self.wbOutputs[0] = inputs
        self._forwardPass()
        rVal = self.wbOutputs[-1]
        self._initializeVars()

        return rVal

    def _fire(self, layer):
        self.wbOutputs[layer] = self.activations[layer](self.weightsList[layer] @ self.wbOutputs[layer - 1] +
                                                        self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        self.weightsList[layer] -= self.deltaWeights[layer]

    def _trainer(self, batch):
        self.wbOutputs[0], self.target = batch
        self._forwardPass()
        loss, self.deltaLoss[-1] = self.lossFunction(self.wbOutputs[-1], self.target)
        self._backPropagate()

        return loss

    def train(self, epochs: "int" = None, batchSize: "int" = None,
              trainDataBase: "DataBase" = None, costFunction: "LossFunction" = None,
              wbOptimizer: "WBOptimizer" = None,
              profile: "bool" = False,
              test: "DataBase" = None):
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
        if trainDataBase is not None:
            self.trainDataBase = trainDataBase
        if costFunction is not None:
            self.lossFunction = costFunction
        if wbOptimizer is not None:
            self.wbOptimizer = wbOptimizer

        self.deltaLoss = [(np.zeros((self.batchSize, self.wbShape[i], 1), dtype=np.float32))
                          for i in range(0, self.wbShape.LAYERS)]
        super(ArtificialNeuralNetwork, self).train(profile=profile, test=test)
