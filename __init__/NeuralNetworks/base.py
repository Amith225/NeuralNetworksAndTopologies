import time
import warnings
import cProfile
import traceback
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..tools import *
    from ..Topologies import *

import numpy as np

from ..tools import MagicBase, metaMagicProperty, Activators, PrintCols, iterable
from ..Topologies import Initializer, LossFunction


class BaseShape(MagicBase, metaclass=metaMagicProperty(ABCMeta)):
    """

    """
    def __str__(self):
        LAYERS = self.LAYERS
        INPUT = self.INPUT
        HIDDEN = self.HIDDEN
        OUTPUT = self.OUTPUT
        return f"{super(BaseShape, self).__str__()[:-1]}:, {LAYERS=}, {INPUT=}, {HIDDEN=}, {OUTPUT=}>"

    def __save__(self):
        pass

    def __init__(self, *shapes):
        """do not change the signature of __init__"""
        self.RAW_SHAPES = shapes
        self.SHAPES = self._formatShapes(shapes)
        assert hash(self.SHAPES)
        self.LAYERS = len(self.SHAPES)
        self.INPUT = self.SHAPES[0]
        self.HIDDEN = self.SHAPES[1:-1]
        self.OUTPUT = self.SHAPES[-1]

    def __getitem__(self, item):
        shapes = self.RAW_SHAPES[item]
        return self.__class__(*shapes) if isinstance(item, slice) and shapes else self.SHAPES[item]

    @staticmethod
    @abstractmethod
    def _formatShapes(shapes) -> tuple:
        """
        method to format given shapes
        :return: hashable formatted shapes
        """


class UniversalShape(BaseShape):
    """
    Allows any shape format, creates 'BaseShape' like object
    """
    @staticmethod
    def _formatShapes(shapes) -> tuple:
        if iterable(shapes):
            assert len(shapes) > 0
            formattedShape = []
            for s in shapes:
                formattedShape.append(UniversalShape._formatShapes(s))
            return tuple(formattedShape)
        else:
            return shapes


class BaseLayer(MagicBase, metaclass=metaMagicProperty(ABCMeta)):
    """

    """
    def __str__(self):
        SHAPE = str(self.SHAPE)
        INITIALIZER = self.INITIALIZER
        OPTIMIZER = self.OPTIMIZER
        DEPS = ', '.join(f"{d}:shape{self.__dict__[d].shape}" for d in self.deps)
        return f"{super(BaseLayer, self).__str__()[:-1]}\n{SHAPE=}\n{INITIALIZER=}\n{OPTIMIZER=}\n{DEPS=}>"

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape", initializer: "Initializer.Base", optimizer: "Optimizer.Base",
                 *depArgs, **depKwargs):
        """
        :param shape: input, output, intermediate(optional) structure of the layer
        """
        self.SHAPE = shape
        self.INITIALIZER = initializer
        self.OPTIMIZER = optimizer

        self.__magic_start__()
        self.input = np.zeros((1, *self.SHAPE[0]), dtype=np.float32)
        self.output = np.zeros((1, *self.SHAPE[-1]), dtype=np.float32)
        self.givenDelta = np.zeros((1, *self.SHAPE[-1]), dtype=np.float32)
        self.delta = np.zeros((1, *self.SHAPE[0]), dtype=np.float32)
        self.deps = self._defineDeps(*depArgs, **depKwargs)
        self.__magic_end__()

    def forPass(self, _input: "np.ndarray"):
        """
        method for forward pass of inputs
        :param _input: self.output of the previous layer
        """
        self.input = _input
        self.output = self._fireAndFindOutput()

    def backProp(self, _delta: "np.ndarray"):
        """
        method for back propagation of deltas
        :param _delta: self.delta of the following layer
        """
        self.givenDelta = _delta
        self.delta = self._wireAndFindDelta()

    def changeOptimizer(self, optimizer):
        self.OPTIMIZER = optimizer
        self._initializeDepOptimizer()

    @abstractmethod
    def _initializeDepOptimizer(self):
        pass

    @abstractmethod
    def _defineDeps(self, *depArgs, **depKwargs) -> list['str']:
        """
        define all dependant objects ($deps) for the layer
        :return: list['str'] -> list of $deps name to be used in __str__
        """

    @abstractmethod
    def _fireAndFindOutput(self) -> "np.ndarray":
        """
        method to use $deps & calculate output (input for next layer)
        :return: np.ndarray -> will be assigned to self.output
        """

    @abstractmethod
    def _wireAndFindDelta(self) -> "np.ndarray":
        """
        method to adjust $deps & calculate delta for the previous layer
        :return: np.ndarray -> will be assigned to self.delta
        """


class BasePlot(MagicBase, metaclass=metaMagicProperty(ABCMeta)):
    """

    """


class BaseNN(MagicBase, metaclass=metaMagicProperty(ABCMeta)):
    """

    """
    SMOOTH_PRINT_INTERVAL = 0.25

    def __str__(self):
        return super(BaseNN, self).__str__()

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Abstract" = None):
        if initializers is None: initializers = Initializers(Initializer.Xavier(2), ..., Initializer.Xavier())
        if activators is None: activators = Activators(ActivationFunction.Prelu(), ..., ActivationFunction.Softmax())
        if lossFunction is None: lossFunction = LossFunction.MeanSquare()

        self.SHAPE = shape
        self.INITIALIZERS = initializers
        self.ACTIVATORS = activators
        self.LOSS_FUNCTION = lossFunction

        self.__magic_start__()
        self.network = self._constructNetwork()

        self.costHistory, self.accuracyHistory = [], []
        self.cost = self.trainAccuracy = self.testAccuracy = float('nan')
        self.costTrained = self.timeTrained = self.epochTrained = 0

        self.numEpochs = self.batchSize = 1
        self.epoch = self.batch = 0
        self.numBatches = None
        self.training = self.profiling = False
        self.trainDataBase = self.testDataBase = self.optimizers = None
        self.__magic_end__()

    @abstractmethod
    def _constructNetwork(self) -> "Network":
        pass

    def process(self, _input) -> "np.ndarray":
        if self.training:
            warnings.showwarning("processing while training in progress may have unintended conflicts",
                                 ResourceWarning, 'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN
        return self.network.forwardPass(np.array(_input))

    def train(self, epochs: int = None,
              batchSize: int = None,
              trainDataBase: "DataBase" = None,
              optimizers: "Optimizers" = None,
              profile: bool = False,
              test: Union[bool, "DataBase"] = None):
        # todo: implement "runs"
        if epochs is not None: self.numEpochs = epochs
        if batchSize is not None: self.batchSize = batchSize
        if trainDataBase is not None: self.trainDataBase = trainDataBase
        if optimizers is not None: self.optimizers = optimizers
        if trainDataBase is not None or batchSize is not None:
            self.numBatches = int(np.ceil(self.trainDataBase.size / self.batchSize))

        self.training = True
        if profile:
            self.profiling = True
            cProfile.run("self.train(test=test)")
            self.profiling = False
        else:
            self._statPrinter('Epoch', f"0/{self.numEpochs}", prefix=PrintCols.CBOLDITALICURL + PrintCols.CBLUE)

            if self.epochTrained == 0:
                loss, _, acc = self._trainer(self.trainDataBase[:self.batchSize])
                trainCosts, trainAccuracy = [loss], [acc]
            else:
                trainCosts, trainAccuracy = [self.costHistory[-1][-1]], [self.accuracyHistory[-1][-1]]
            trainTime = 0
            nextPrintTime = self.SMOOTH_PRINT_INTERVAL
            for self.epoch in range(1, self.numEpochs + 1):
                self.costTrained = self.trainAccuracy = 0
                timeStart = time.time()
                batchGenerator = self.trainDataBase.batchGenerator(self.batchSize)
                try:
                    for self.batch, _batch in enumerate(batchGenerator):
                        loss, delta, acc = self._trainer(_batch)
                        self.costTrained += loss
                        self.trainAccuracy += acc
                        self.network.backPropagation(delta)
                    self.epochTrained += 1
                except Exception:  # noqa
                    traceback.print_exc()
                    warnings.showwarning("unhandled exception occurred while training,"
                                         "\nquiting training and rolling back to previous auto save", RuntimeWarning,
                                         'base.py', 0)
                    NotImplemented  # todo: roll back and auto save (P.S save also)
                    return -1
                epochTime = time.time() - timeStart
                trainTime += epochTime
                self.costTrained /= self.trainDataBase.size
                self.trainAccuracy /= self.numBatches
                trainCosts.append(self.costTrained)
                trainAccuracy.append(self.trainAccuracy)
                if trainTime >= nextPrintTime or self.epoch == self.numEpochs:
                    nextPrintTime += self.SMOOTH_PRINT_INTERVAL
                    avgTime = trainTime / self.epoch
                    print(end='\r')
                    self._statPrinter('Epoch', f"{self.epoch}/{self.numEpochs}",
                                      prefix=PrintCols.CBOLDITALICURL + PrintCols.CBLUE)
                    self._statPrinter('Cost', f"{self.costTrained:.4f}",
                                      prefix=PrintCols.CYELLOW, suffix='')
                    self._statPrinter('Cost-Reduction', f"{(trainCosts[-2] - self.costTrained):.4f}")
                    self._statPrinter('Accuracy', f"{self.trainAccuracy:.4f}",
                                      prefix=PrintCols.CYELLOW, suffix='')
                    self._statPrinter('Accuracy-Increment', f"{(self.trainAccuracy - trainAccuracy[-2]):.4f}")
                    self._statPrinter('Time', self.secToHMS(epochTime),
                                      prefix=PrintCols.CBOLD + PrintCols.CRED2, suffix='')
                    self._statPrinter('Average-Time', self.secToHMS(avgTime),
                                      suffix='')
                    self._statPrinter('Eta', self.secToHMS(avgTime * (self.numEpochs - self.epoch)),
                                      suffix='')
                    self._statPrinter('Elapsed', self.secToHMS(trainTime))
            self._statPrinter('', '', end='\n')
            self.timeTrained += trainTime
            self.costHistory.append(trainCosts)
            self.accuracyHistory.append(trainAccuracy)

            if test or test is None: self.test(test)
        self.training = False

    def _trainer(self, _batch) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        output, target = self.network.forwardPass(_batch[0]), _batch[1]
        loss, delta = self.LOSS_FUNCTION(output, target)
        acc = self._tester(output, target)
        return loss, delta, acc

    def test(self, testDataBase: "DataBase" = None):
        self._statPrinter('Testing', 'wait...', prefix=PrintCols.CBOLD + PrintCols.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            self.trainAccuracy = self.accuracy(self.trainDataBase.inputSet, self.trainDataBase.targetSet)
        if testDataBase is not None: self.testAccuracy = self.accuracy(testDataBase.inputSet, testDataBase.targetSet)
        print(end='\r')
        self._statPrinter('Train-Accuracy', f"{self.trainAccuracy}%", suffix='', end='\n')
        self._statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')

    @staticmethod
    def _tester(_output, _target) -> "np.ndarray":
        if np.shape(_target) != 1:
            # poly node multi classification
            outIndex = np.where(_output == np.max(_output, axis=1, keepdims=True))[1]
            targetIndex = np.where(_target == 1)[1]
        else:
            # single node binary classification
            outIndex = np.round(_output)
            targetIndex = _target
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

    @staticmethod
    def _statPrinter(key, value, *, prefix='', suffix=PrintCols.CEND, end=' '):
        print(prefix + f"{key}:{value}" + suffix, end=end)

    @staticmethod
    def secToHMS(seconds, hms=('h', 'm', 's')):
        encode = f'%S{hms[2]}'
        if (tim := time.gmtime(seconds)).tm_min != 0: encode = f'%M{hms[1]}' + encode
        if tim.tm_hour != 0: encode = f'%H{hms[0]}' + encode

        return time.strftime(encode, tim)
