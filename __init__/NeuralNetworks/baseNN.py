import warnings as wr
import time as tm
import cProfile as cP
from abc import ABCMeta, abstractmethod
from typing import *

if TYPE_CHECKING:
    from ..tools import *
    from ..Topologies import *

import numpy as np

from ..tools import *
pV = PrintVars


class AbstractNN(AbstractSave, AbstractLoad, metaclass=ABCMeta):
    DEFAULT_DIR = '\\Models\\'
    DEFAULT_NAME = 'nn'
    FILE_TYPE = '.nnt'
    SMOOTH_PRINT_INTERVAL = 0.25

    def saveName(self) -> str:
        pass

    def _write(self, dumpFile, *args, **kwargs) -> str:
        pass

    @classmethod
    def _read(cls, loadFile, *args, **kwargs):
        pass

    def __init__(self,
                 shape: "Shape",
                 initializer: "AbstractInitializer",
                 activators: "Activators",
                 lossFunction: "AbstractLossFunction"):
        self.shape = shape
        self.initializer = initializer
        self.activations, self.activationDerivatives = activators(self.shape.LAYERS - 1)
        self.lossFunction = lossFunction

        self.costHistory, self.accuracyHistory = [], []
        self.costTrained = self.timeTrained = self.epochTrained = 0
        self.trainAccuracy = self.testAccuracy = float('nan')

        self.epochs, self.batchSize = 1, 32
        self.numBatches = None
        self.training = self.profiling = False
        self.__neverTrained = True

        self.outputs = self.target = self.deltaLoss = self.loss = None
        self.__deInitializeTrainDep()

        self.trainDataBase = self.testDataBase = None
        self.optimizer = None

        self.outputShape = self._findOutputShape()
        self._initializeWeightsBiasesDelta()

    @abstractmethod
    def _forwardPass(self, layer=1):
        pass

    @abstractmethod
    def _backPropagate(self, layer=-1):
        pass

    @abstractmethod
    def _fire(self, layer):
        pass

    @abstractmethod
    def _wire(self, layer):
        pass

    @abstractmethod
    def _findOutputShape(self):
        pass

    @abstractmethod
    def _initializeWeightsBiasesDelta(self):
        pass

    def process(self, inputs) -> "np.ndarray":
        if self.training:
            wr.showwarning("processing while training in progress, may have unintended conflicts", ResourceWarning,
                           'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN

        inputs = np.array(inputs).reshape([-1, *self.shape.INPUT])
        self.__deInitializeTrainDep()
        self.outputs[0] = inputs
        self._forwardPass()
        out = self.outputs[-1]
        self.__deInitializeTrainDep()

        return out

    def __initializeTrainDep(self):
        self.outputs = [np.zeros((self.trainDataBase.size, *self.outputShape[i]), dtype=np.float32)
                        for i in range(self.shape.LAYERS)]
        self.target = np.zeros((self.trainDataBase.size, *self.outputShape[-1]), dtype=np.float32)
        self.deltaLoss = [(np.zeros((self.batchSize, *self.shape[i]), dtype=np.float32))
                          for i in range(self.shape.LAYERS)]
        self.loss = 0

    def __deInitializeTrainDep(self):
        self.outputs = [None for _ in range(self.shape.LAYERS)]
        self.target = None
        self.deltaLoss = [None for _ in range(self.shape.LAYERS)]
        self.loss = None

    def train(self,
              epochs: "int",
              batchSize: "int",
              trainDataBase: "DataBase",
              optimizer,
              profile=False,
              test=None):
        if epochs is not None: self.epochs = epochs
        if batchSize is not None: self.batchSize = batchSize
        if trainDataBase is not None: self.trainDataBase = trainDataBase
        if optimizer is not None: self.optimizer = optimizer

        if not profile:
            if self.__neverTrained:
                self.__deInitializeTrainDep()
                self.outputs[0], self.target = self.trainDataBase.batchGenerator(self.batchSize).send(-1)
                self._forwardPass()
                self.loss, self.deltaLoss[-1] = self._trainer()
                trainCosts, trainAccuracy = [self.loss], [self._tester(self.outputs[-1], self.target)]
                self.__neverTrained = False
            else:
                trainCosts, trainAccuracy = [self.costHistory[-1][-1]], [self.accuracyHistory[-1][-1]]

            self.__initializeTrainDep()
            self.training = True
            self.numBatches = int(np.ceil(self.trainDataBase.size / self.batchSize))
            trainTime = 0
            nextPrintTime = self.SMOOTH_PRINT_INTERVAL
            self._statPrinter('Epoch', f"0/{self.epochs}", prefix=pV.CBOLDITALICURL + pV.CBLUE)
            for epoch in range(1, self.epochs + 1):
                self.costTrained = self.trainAccuracy = 0
                time = tm.time()
                batchGenerator = self.trainDataBase.batchGenerator(self.batchSize)
                for batch in range(self.numBatches):
                    self.outputs[0], self.target = batchGenerator.__next__()
                    self._forwardPass()
                    self.loss, self.deltaLoss[-1] = self._trainer()
                    self._backPropagate()
                    self.costTrained += self.loss
                    self.trainAccuracy += self._tester(self.outputs[-1], self.target)
                epochTime = tm.time() - time
                trainTime += epochTime
                self.costTrained /= self.trainDataBase.size
                self.trainAccuracy /= self.numBatches
                trainCosts.append(self.costTrained)
                trainAccuracy.append(self.trainAccuracy)
                if trainTime >= nextPrintTime or epoch == self.epochs:
                    nextPrintTime += self.SMOOTH_PRINT_INTERVAL
                    avgTime = trainTime / epoch
                    print(end='\r')
                    self._statPrinter('Epoch', f"{epoch}/{self.epochs}", prefix=pV.CBOLDITALICURL + pV.CBLUE)
                    self._statPrinter('Cost', f"{self.costTrained:.4f}", prefix=pV.CYELLOW, suffix='')
                    self._statPrinter('Cost-Reduction', f"{(trainCosts[-2] - self.costTrained):.4f}")
                    self._statPrinter('Accuracy', f"{self.trainAccuracy:.4f}", prefix=pV.CYELLOW, suffix='')
                    self._statPrinter('Accuracy-Increment', f"{(self.trainAccuracy - trainAccuracy[-2]):.4f}")
                    self._statPrinter('Time', self.secToHMS(epochTime), prefix=pV.CBOLD + pV.CRED2, suffix='')
                    self._statPrinter('Average-Time', self.secToHMS(avgTime), suffix='')
                    self._statPrinter('Eta', self.secToHMS(avgTime * (self.epochs - epoch)), suffix='')
                    self._statPrinter('Elapsed', self.secToHMS(trainTime))
                self.epochTrained += 1
            self._statPrinter('', '', end='\n')
            self.timeTrained += trainTime
            self.costHistory.append(trainCosts)
            self.accuracyHistory.append(trainAccuracy)
            self.training = False
            self.__deInitializeTrainDep()
        else:
            self.profiling = True
            cP.runctx("self.train()", globals=globals(), locals=locals())
            self.profiling = False

        if not self.profiling: self.test(test)

    def _trainer(self):
        return self.lossFunction(self.outputs[-1], self.target)

    def test(self, testDataBase: "DataBase" = None):
        self._statPrinter('Testing', 'wait...', prefix=pV.CBOLD + pV.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            self.trainAccuracy = self.accuracy(self.trainDataBase.inputSet, self.trainDataBase.targetSet)
        if testDataBase is not None: self.testAccuracy = self.accuracy(testDataBase.inputSet, testDataBase.targetSet)
        print(end='\r')
        self._statPrinter('Train-Accuracy', f"{self.trainAccuracy}%", suffix='', end='\n')
        self._statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')

    def accuracy(self, inputSet, targetSet):
        assert (size := np.shape(inputSet)[0]) == np.shape(targetSet)[0], \
            "the size of both inputSet and targetSet should be same"
        try:
            return self._tester(self.process(inputSet), targetSet)
        except MemoryError:
            accuracy1 = self.accuracy(inputSet[:(to := size // 2)], targetSet[:to])
            accuracy2 = self.accuracy(inputSet[to:], targetSet[to:])
            return (accuracy1 + accuracy2) / 2

    def _tester(self, out, tar):
        if np.shape(tar) != 1:
            outIndex = np.where(out == np.max(out, axis=1, keepdims=True))[1]
            targetIndex = np.where(tar == 1)[1]
        else:
            outIndex = np.round(out)
            targetIndex = tar
        result = outIndex == targetIndex
        result = np.float32(1) * result

        return result.sum() / result.shape[0] * 100

    @staticmethod
    def _statPrinter(key, value, *, prefix='', suffix=pV.CEND, end=' '):
        print(prefix + f"{key}:{value}" + suffix, end=end)

    @staticmethod
    def secToHMS(seconds, hms=('h', 'm', 's')):
        encode = f'%S{hms[2]}'
        if (tim := tm.gmtime(seconds)).tm_min != 0:
            encode = f'%M{hms[1]}' + encode
        if tim.tm_hour != 0:
            encode = f'%H{hms[0]}' + encode

        return tm.strftime(encode, tim)
