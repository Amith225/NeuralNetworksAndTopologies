import time
import warnings
import cProfile
import traceback
import pstats
import os
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..tools import *
    from ..Topologies import *

import numpy as np

from ..tools import MagicProperty, makeMetaMagicProperty, \
    PrintCols, iterable, secToHMS, statPrinter
from ..Topologies import Activators, Initializers, Optimizers, LossFunction, DataBase


class BaseShape(metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.NUM_LAYERS}:{self.SHAPES}>"

    def __save__(self):
        pass

    def __getitem__(self, item):
        shapes = self.RAW_SHAPES[item]
        return self.__class__(*shapes) if isinstance(item, slice) and shapes else self.SHAPES[item]

    def __hash__(self):
        return hash(self.SHAPES)

    def __init__(self, inputShape, *shapes):
        self.RAW_SHAPES = inputShape, *shapes
        self.SHAPES = self._formatShapes(self.RAW_SHAPES)
        assert hash(self.SHAPES)
        self.NUM_LAYERS = len(self.SHAPES)
        self.INPUT = self.SHAPES[0]
        self.HIDDEN = self.SHAPES[1:-1]
        self.OUTPUT = self.SHAPES[-1]

    @staticmethod
    @abstractmethod
    def _formatShapes(shapes) -> tuple:
        """
        method to format given shapes
        :return: hashable formatted shapes
        """


class UniversalShape(BaseShape):
    """Allows any shape format, creates 'BaseShape' like object"""

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


class BaseLayer(metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.SHAPE}: Ini={self.INITIALIZER}: Opt={self.optimizer}: " \
               f"AF={self.ACTIVATION_FUNCTION}>"

    def __str__(self):
        DEPS = ': '.join(f"{dName}:shape{getattr(self, dName).shape}" for dName in self.DEPS)
        return f"{self.__repr__()[:-1]}:\n{DEPS=}>"

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape",
                 initializer: "Initializers.Base",
                 optimizer: "Optimizers.Base",
                 activationFunction: "Activators.Base",
                 *depArgs, **depKwargs):
        """
        :param shape: input, output, intermediate(optional) structure of the layer
        """
        self.SHAPE = shape
        self.INITIALIZER = initializer
        self.optimizer = optimizer
        self.ACTIVATION_FUNCTION = activationFunction

        self.input = np.zeros(self.SHAPE[0], dtype=np.float32)
        self.output = np.zeros(self.SHAPE[-1], dtype=np.float32)
        self.inputDelta = np.zeros(self.SHAPE[-1], dtype=np.float32)
        self.outputDelta = np.zeros(self.SHAPE[0], dtype=np.float32)

        self.DEPS = self._defineDeps(*depArgs, **depKwargs)

    def forPass(self, _input: "np.ndarray") -> "np.ndarray":
        f"""
        method for forward pass of inputs
        :param _input: self.output from the lower layer
        :return: {self.output}
        """
        self.input = _input
        self.output = self._fire()
        return self.output

    def backProp(self, _delta: "np.ndarray") -> "np.ndarray":
        f"""
        method for back propagation of deltas
        :param _delta: value for {self.inputDelta} from {self.outputDelta} of the higher layer
        :return: {self.outputDelta}
        """
        self.inputDelta = _delta
        self.outputDelta = self._wire()
        return self.outputDelta

    def changeOptimizer(self, optimizer: "Optimizers.Base"):
        self.optimizer = optimizer
        self._initializeDepOptimizer()

    @abstractmethod
    def _initializeDepOptimizer(self):
        f"""create new optimizer instance for each dep in {self.DEPS} by using {self.optimizer.__new_copy__()}"""

    @abstractmethod
    def _defineDeps(self, *depArgs, **depKwargs) -> list['str']:
        f"""
        define all dependant objects for the layer
        :return: value for {self.DEPS}
        """

    @abstractmethod
    def _fire(self) -> "np.ndarray":
        f"""
        :return: value for {self.output}, is input for the higher layer
        """

    @abstractmethod
    def _wire(self) -> "np.ndarray":
        f"""
        :return: value for {self.outputDelta}, is delta for the lower layer
        """


class BasePlot(metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """


class Network:
    """

    """

    def __repr__(self):
        LossFunction = self.LOSS_FUNCTION  # noqa
        return f"{self.__class__.__name__}:{LossFunction=}"

    def __str__(self):
        layers = "\n\t\t".join(repr(layer) for layer in self.LAYERS)
        return f"{super(Network, self).__str__()}:\n\t\t{layers}"

    def __save__(self):
        pass

    def __init__(self, inputLayer: "BaseLayer", *layers: "BaseLayer", lossFunction: "LossFunction.Base"):
        assert len(layers) > 0
        self.LAYERS = inputLayer, *layers
        self.INPUT_LAYER = inputLayer
        self.HIDDEN_LAYERS = layers[:-1]
        self.OUTPUT_LAYER = layers[-1]
        self.LOSS_FUNCTION = lossFunction

    def changeOptimizer(self, _optimizer: Union["Optimizers.Base", "Optimizers"], index: int = None):
        f"""
        changes optimizer at index if given else changes all the optimizers to {_optimizer} or 
        uses given collection {Optimizers}
        """
        assert isinstance(_optimizer, (Optimizers, Optimizers.Base))
        if index is None:
            optimizers = _optimizer.get(len(self.LAYERS)) if isinstance(_optimizer, Optimizers) else \
                (_optimizer,) * len(self.LAYERS)
            for i, layer in enumerate(self.LAYERS):
                layer.changeOptimizer(optimizers[i])
        else:
            layer: "BaseLayer" = self.LAYERS[index]
            layer.changeOptimizer(_optimizer)

    def forwardPass(self, _input) -> "np.ndarray":
        f"""
        calls(and sends hierarchical I/O) the forPass method of all the layers
        :param _input: input for {self.INPUT_LAYER}
        :return: output of {self.OUTPUT_LAYER}
        """
        _output = self.INPUT_LAYER.forPass(_input)
        for layer in self.HIDDEN_LAYERS: _output = layer.forPass(_output)
        return self.OUTPUT_LAYER.forPass(_output)

    def backPropagation(self, _delta) -> "np.ndarray":
        f"""
        calls(and sends hierarchical I/O) the backProp method of all the layers
        :param _delta: delta for {self.OUTPUT_LAYER}
        :return: delta of {self.INPUT_LAYER}
        """
        _delta = self.OUTPUT_LAYER.backProp(_delta)
        for reversedLayer in self.HIDDEN_LAYERS[::-1]: _delta = reversedLayer.backProp(_delta)
        return self.INPUT_LAYER.backProp(_delta)


class BaseNN(metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """
    STAT_PRINT_INTERVAL = 1
    __optimizers = Optimizers(Optimizers.Adam(), ..., Optimizers.AdaGrad())

    @MagicProperty
    def optimizers(self):
        return self.__optimizers

    @optimizers.setter
    def optimizers(self, _optimizers: "Optimizers"):
        self.__optimizers = _optimizers
        self.NETWORK.changeOptimizer(self.__optimizers)

    def __repr__(self):
        Shape = self.SHAPE
        Cost, Time, Epochs = self.costTrained, secToHMS(self.timeTrained), self.epochTrained
        acc = int(self.testAccuracy), int(self.accuracyTrained)
        return f"<{self.__class__.__name__}:Acc={acc[0]}%,{acc[1]}%: {Cost=:07.4f}: {Time=}: {Epochs=}: {Shape=}>"

    def __str__(self):
        Optimizers = self.optimizers  # noqa
        TrainDataBase, TestDataBase = self.trainDataBase, self.testDataBase
        return f"{self.__repr__()[1:-1]}:\n\t{Optimizers=}\n\t{TrainDataBase=}\n\t{TestDataBase=}\n\t{self.NETWORK}"

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Base" = None, *ntwArgs, **ntwKwargs):
        if initializers is None: initializers = Initializers(Initializers.Xavier(2), ..., Initializers.Xavier())
        if activators is None: activators = Activators(Activators.PRelu(), ..., Activators.SoftMax())
        if lossFunction is None: lossFunction = LossFunction.MeanSquare()

        self.SHAPE = shape

        self.costHistory, self.accuracyHistory = [], []
        self.accuracyTrained = self.testAccuracy = 0
        self.costTrained = self.timeTrained = self.epochTrained = 0

        self.numEpochs = self.batchSize = 1
        self.epoch = self.batch = 0
        self.numBatches = None
        self.training = self.profiling = False
        self.trainDataBase = self.testDataBase = None

        self.NETWORK = self._constructNetwork(initializers, activators, lossFunction, *ntwArgs, **ntwKwargs)

    @abstractmethod
    def _constructNetwork(self, initializers: "Initializers" = None,
                          activators: "Activators" = None,
                          lossFunction: "LossFunction.Base" = None, *args, **kwargs) -> "Network":
        pass

    def process(self, _input) -> "np.ndarray":
        if self.training:
            warnings.showwarning("processing while training in progress may have unintended conflicts",
                                 ResourceWarning, 'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN
        return self.NETWORK.forwardPass(np.array(_input))

    def profile(self):
        self.profiling = True
        prof = cProfile.Profile()
        prof.runctx("self._train()", locals=locals(), globals=globals())
        prof.print_stats('cumtime')
        prof.dump_stats('profile.pstat')
        with open('profile.txt', 'w') as stream:
            stats = pstats.Stats('profile.pstat', stream=stream)
            stats.sort_stats('cumtime')
            stats.print_stats()
        os.remove('profile.txt')
        self.profiling = False

    def _train(self):
        statPrinter('Epoch', f"0/{self.numEpochs}", prefix=PrintCols.CBOLDITALICURL + PrintCols.CBLUE)
        self.training = True
        if self.epochTrained == 0:
            loss, _, acc = self._trainer(self.trainDataBase[:self.batchSize])
            trainCosts, trainAccuracies = [loss], [acc]
        else:
            trainCosts, trainAccuracies = [self.costHistory[-1][-1]], [self.accuracyHistory[-1][-1]]

        for self.epoch in range(1, self.numEpochs + 1):
            epochTime = nextPrintTime = 0
            costTrained = accuracyTrained = 0
            try:
                for self.batch, _batch in enumerate(self.trainDataBase.batchGenerator(self.batchSize)):
                    timeStart = time.time()
                    loss, delta, acc = self._trainer(_batch)
                    self.NETWORK.backPropagation(delta)
                    costTrained += loss
                    accuracyTrained += acc
                    batchTime = time.time() - timeStart
                    epochTime += batchTime
                    if epochTime >= nextPrintTime or self.batch == self.numBatches - 1:
                        nextPrintTime += self.STAT_PRINT_INTERVAL
                        self.printStats(costTrained / (self.batch + 1), trainCosts[-1],
                                        accuracyTrained / (self.batch + 1), trainAccuracies[-1], epochTime)
                self.timeTrained += epochTime
                self.epochTrained += 1
                self.costTrained = costTrained / self.numBatches
                self.accuracyTrained = accuracyTrained / self.numBatches
                trainCosts.append(self.costTrained)
                trainAccuracies.append(self.accuracyTrained)
            except Exception:  # noqa
                traceback.print_exc()
                warnings.showwarning("unhandled exception occurred while training,"
                                     "\nquiting training and rolling back to previous auto save", RuntimeWarning,
                                     'base.py', 0)
                raise NotImplementedError  # todo: roll back and auto save
        self.costHistory.append(trainCosts)
        self.accuracyHistory.append(trainAccuracies)
        self.training = False
        statPrinter('', '', end='\n')

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
        assert isinstance(self.trainDataBase, DataBase)
        assert self.trainDataBase.inpShape == self.SHAPE.INPUT and self.trainDataBase.tarShape == self.SHAPE.OUTPUT
        if trainDataBase is not None or batchSize is not None:
            self.numBatches = int(np.ceil(self.trainDataBase.size / self.batchSize))
        if profile:
            self.profile()
        else:
            self._train()
        if test or test is None: self.test(test)

    def printStats(self, loss, prevLoss, acc, prevAcc, epochTime):
        print(end='\r')
        """__________________________________________________________________________________________________________"""
        statPrinter('Epoch', f"{self.epoch:0{len(str(self.numEpochs))}d}/{self.numEpochs}",
                    prefix=PrintCols.CBOLDITALICURL + PrintCols.CBLUE, suffix='')
        statPrinter('Batch', f"{(b := self.batch + 1):0{len(str(self.numBatches))}d}/{self.numBatches}",
                    suffix='', end='')
        statPrinter(f"({int(b / self.numBatches * 100):03d}%)", '')
        """__________________________________________________________________________________________________________"""
        statPrinter('Cost', f"{loss:07.4f}", prefix=PrintCols.CYELLOW, suffix='')
        statPrinter('Cost-Dec', f"{(prevLoss - loss):07.4f}", suffix='')
        statPrinter('Acc', f"{int(acc):03d}%", prefix=PrintCols.CYELLOW, suffix='')
        statPrinter('Acc-Inc', f"{int(acc - prevAcc):03d}%")
        """__________________________________________________________________________________________________________"""
        elapsed = self.timeTrained + epochTime
        avgTime = elapsed / (effectiveEpoch := self.epoch - 1 + (self.batch + 1) / self.numBatches)
        statPrinter('Time', secToHMS(elapsed), prefix=PrintCols.CBOLD + PrintCols.CRED2, suffix='')
        statPrinter('Epoch-Time', secToHMS(epochTime), suffix='')
        statPrinter('Avg-Time', secToHMS(avgTime), suffix='')
        statPrinter('Eta', secToHMS(avgTime * (self.numEpochs - effectiveEpoch)))
        """__________________________________________________________________________________________________________"""

    def _trainer(self, _batch: tuple["np.ndarray", "np.ndarray"]):
        output, target = self.NETWORK.forwardPass(_batch[0]), _batch[1]
        loss, delta = self.NETWORK.LOSS_FUNCTION(output, target)
        acc = self._tester(output, target)
        return loss, delta, acc

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
        statPrinter('Testing', 'wait...', prefix=PrintCols.CBOLD + PrintCols.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            self.accuracyTrained = self.accuracy(self.trainDataBase.inputs, self.trainDataBase.targets)
        if testDataBase is not None:
            assert isinstance(testDataBase, DataBase)
            assert testDataBase.inpShape == self.SHAPE.INPUT and testDataBase.tarShape == self.SHAPE.OUTPUT
            self.testAccuracy = self.accuracy(testDataBase.inputs, testDataBase.targets)
        print(end='\r')
        statPrinter('Train-Accuracy', f"{self.accuracyTrained}%", suffix='', end='\n')
        statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')

    @staticmethod
    def _tester(_output: "np.ndarray", _target: "np.ndarray") -> "np.ndarray":
        if np.shape(_target) != 1:
            # poly node multi classification
            outIndex = np.argmax(_output, axis=1)
            targetIndex = np.argmax(_target, axis=1)
        else:
            # single node binary classification
            outIndex = _output.round()
            targetIndex = _target
        result: "np.ndarray" = outIndex == targetIndex
        return result.mean() * 100
