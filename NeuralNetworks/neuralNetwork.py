import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..Topologies import DataBase, LossFunction, Initializer
    from ..Utils import Shape, Activators
import cProfile as cP
import time as tm
import warnings as wr
from abc import ABCMeta, abstractmethod

import dill as dl
import numpy as np

from ._printVars import PrintVars as pV
from Utils import AbstractSave


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

    def __init__(self, shape: "Shape", activators: "Activators", costFunction: "LossFunction"):
        self.shape = shape
        self.activations, self.activationDerivatives = activators(self.shape.LAYERS - 1)
        self.lossFunction = costFunction

        self.costHistory = []
        self.accuracyHistory = []
        self.costTrained = 0
        self.timeTrained = 0
        self.epochTrained = 0
        self.trainAccuracy = float('nan')
        self.testAccuracy = float('nan')

        self.epochs = 1
        self.batchSize = 32
        self.numBatches = None
        self.training = False
        self.profiling = False
        self.__neverTrained = True

        self.outputs = None
        self.target = None
        self.deltaLoss = None
        self.loss = None

        self.trainDataBase = None
        self.optimizer = None

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
    def _initializeVars(self):
        pass

    @abstractmethod
    def _trainer(self):
        """assign self.loss and self.deltaLoss here"""

    def process(self, inputs) -> "np.ndarray":
        if self.training:
            wr.showwarning("processing while training in progress, may have unintended conflicts", ResourceWarning,
                           'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN

        inputs = np.array(inputs).reshape([-1, *self.shape.INPUT])
        self.outputs[0] = inputs
        self._forwardPass()
        rVal = self.outputs[-1]
        self._initializeVars()

        return rVal

    @staticmethod
    def _statPrinter(key, value, prefix='', suffix=pV.CEND, end=' '):
        print(prefix + f"{key}:{value}" + suffix, end=end)

    def train(self, epochs: "int",
              batchSize: "int",
              trainDataBase: "DataBase",
              optimizer,
              profile=False,
              test=None):
        if epochs is not None:
            self.epochs = epochs
        if batchSize is not None:
            self.batchSize = batchSize
        if trainDataBase is not None:
            self.trainDataBase = trainDataBase
        if optimizer is not None:
            self.optimizer = optimizer

        self.deltaLoss = [(np.zeros((self.batchSize, *self.shape[i]), dtype=np.float32))
                          for i in range(0, self.shape.LAYERS)]

        if not profile:
            if self.__neverTrained:
                self.outputs[0], self.target = self.trainDataBase.batchGenerator(self.batchSize).send(-1)
                self._forwardPass(); self._trainer(); self._backPropagate()
                trainCosts, trainAccuracy = [self.loss], [self._tester(self.outputs[-1], self.target)]
                self._initializeVars()
                self.__neverTrained = False
            else: trainCosts, trainAccuracy = [self.costHistory[-1][-1]], [self.accuracyHistory[-1][-1]]
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
                    self._forwardPass(); self._trainer(); self._backPropagate()
                    self.costTrained += self.loss; self.trainAccuracy += self._tester(self.outputs[-1], self.target)
                epochTime = tm.time() - time; trainTime += epochTime
                self.costTrained /= self.trainDataBase.size; self.trainAccuracy /= self.numBatches
                trainCosts.append(self.costTrained); trainAccuracy.append(self.trainAccuracy)
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
            self.costHistory.append(trainCosts); self.accuracyHistory.append(trainAccuracy)
            self.training = False
            self._initializeVars()
        else:
            self.profiling = True; cP.runctx("self.train()", globals=globals(), locals=locals()); self.profiling = False

        if not self.profiling: self.test(test)

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
            self.trainAccuracy = self.accuracy(self.trainDataBase.inputSet, self.trainDataBase.targetSet)
        if testDataBase is not None:
            self.testAccuracy = self.accuracy(testDataBase.inputSet, testDataBase.targetSet)
        print(end='\r')
        self._statPrinter('Train-Accuracy', f"{self.trainAccuracy}%", suffix='', end='\n')
        self._statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')
        self._initializeVars()


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
