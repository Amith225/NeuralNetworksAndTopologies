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

from ..tools import MagicBase, MagicProperty, makeMetaMagicProperty, \
    PrintCols, iterable, secToHMS, statPrinter
from ..Topologies import Activators, Initializers, Optimizers, LossFunction, DataBase


class BaseShape(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
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

    def __getitem__(self, item):
        shapes = self.RAW_SHAPES[item]
        return self.__class__(*shapes) if isinstance(item, slice) and shapes else self.SHAPES[item]

    def __hash__(self):
        return hash(self.SHAPES)

    def __init__(self, *shapes):
        """do not change the signature of __init__"""
        self.RAW_SHAPES = shapes
        self.SHAPES = self._formatShapes(shapes)
        assert hash(self.SHAPES)
        self.LAYERS = len(self.SHAPES)
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


class BaseLayer(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """

    def __str__(self):
        SHAPE = str(self.SHAPE)
        INITIALIZER = self.INITIALIZER
        OPTIMIZER = self.optimizer
        DEPS = ', '.join(f"{dName}:shape{getattr(self, dName).shape}" for dName in self.DEPS)
        return f"{super(BaseLayer, self).__str__()[:-1]}:\n{SHAPE=}\n{INITIALIZER=}\n{OPTIMIZER=}\n{DEPS=}>"

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

        self.input = np.zeros((1, *self.SHAPE[0]), dtype=np.float32)
        self.output = np.zeros((1, *self.SHAPE[-1]), dtype=np.float32)
        self.inputDelta = np.zeros((1, *self.SHAPE[-1]), dtype=np.float32)
        self.outputDelta = np.zeros((1, *self.SHAPE[0]), dtype=np.float32)

        self.DEPS = self._defineDeps(*depArgs, **depKwargs)

    def forPass(self, _input: "np.ndarray") -> "np.ndarray":
        """
        method for forward pass of inputs
        :param _input: self.output from the lower layer
        :return: self.output
        """
        self.input = _input
        self.output = self._fire()
        return self.output

    def backProp(self, _delta: "np.ndarray") -> "np.ndarray":
        """
        method for back propagation of deltas
        :param _delta: self.outputDelta from the higher layer
        :return: self.outputDelta
        """
        self.inputDelta = _delta
        self.outputDelta = self._wire()
        return self.outputDelta

    def changeOptimizer(self, optimizer: "Optimizers.Base"):
        self.optimizer = optimizer
        self._initializeDepOptimizer()

    @abstractmethod
    def _initializeDepOptimizer(self):
        """create new optimizer instance for each dep in $DEPS by using self.optimizer.__new_copy__()"""

    @abstractmethod
    def _defineDeps(self, *depArgs, **depKwargs) -> list['str']:
        """
        define all dependant objects ($DEPS) for the layer
        :return: list of $DEPS whose shape is displayed in __str__
        """

    @abstractmethod
    def _fire(self) -> "np.ndarray":
        """
        method to use $DEPS & calculate output (input for next layer)
        :return: value for self.output
        """

    @abstractmethod
    def _wire(self) -> "np.ndarray":
        """
        method to adjust $DEPS & calculate delta for the lower layer
        :return: value for self.outputDelta
        """


class BasePlot(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """


class Network:
    """

    """
    def __str__(self):
        return super(Network, self).__str__()

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
        if index is None:
            optimizers = _optimizer.get(len(self.LAYERS))
            for i, layer in enumerate(self.LAYERS):
                layer.changeOptimizer(optimizers[i])
        else:
            layer: "BaseLayer" = self.LAYERS[index]
            layer.changeOptimizer(_optimizer)

    def forwardPass(self, _input) -> "np.ndarray":
        _output = self.INPUT_LAYER.forPass(_input)
        for layer in self.HIDDEN_LAYERS: _output = layer.forPass(_output)
        return self.OUTPUT_LAYER.forPass(_output)

    def backPropagation(self, _delta) -> "np.ndarray":
        _delta = self.OUTPUT_LAYER.backProp(_delta)
        for reversedLayer in self.HIDDEN_LAYERS[::-1]: _delta = reversedLayer.backProp(_delta)
        return self.INPUT_LAYER.backProp(_delta)


class BaseNN(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """

    STAT_PRINT_INTERVAL = 1
    __optimizers = Optimizers(Optimizers.AdaGrad(), ...)

    @MagicProperty
    def optimizers(self):
        return self.__optimizers

    @optimizers.setter
    def optimizers(self, _optimizers: "Optimizers"):
        self.__optimizers = _optimizers
        self.NETWORK.changeOptimizer(self.__optimizers)

    def __str__(self):
        SHAPE = str(self.SHAPE)
        INITIALIZERS = str(self.INITIALIZERS)
        ACTIVATORS = str(self.ACTIVATORS)
        OPTIMIZERS = str(self.optimizers)  # noqa
        LOSS_FUNCTION = self.LOSS_FUNCTION
        ACCURACY = {'TRAIN': self.accuracyTrained, 'TEST': self.testAccuracy}
        NumEpochs, NumBatches, BatchSize = self.numEpochs, self.numBatches, self.batchSize
        TRAINED = {'COST': self.costTrained, 'TIME': secToHMS(self.timeTrained), 'EPOCH': self.epochTrained,
                   'RECENT': f"{NumEpochs=}, {NumBatches=}, {BatchSize=}"}
        TRAIN_DATA_BASE, TEST_DATA_BASE = self.trainDataBase, self.testDataBase
        NETWORK = self.NETWORK
        return f"{super(BaseNN, self).__str__()[:-1]}:\n{SHAPE=}\n{INITIALIZERS=}\n{ACTIVATORS=}\n{OPTIMIZERS=}\n" \
               f"{LOSS_FUNCTION=}\n{ACCURACY=}\n{TRAINED=}\n{TRAIN_DATA_BASE=}\n{TEST_DATA_BASE=}\n{NETWORK=}>"

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Base" = None):
        if initializers is None: initializers = Initializers(Initializers.Xavier(2), ..., Initializers.Xavier())
        if activators is None: activators = Activators(Activators.Prelu(), ..., Activators.Softmax())
        if lossFunction is None: lossFunction = LossFunction.MeanSquare()

        self.SHAPE = shape
        self.INITIALIZERS = initializers
        self.ACTIVATORS = activators
        self.LOSS_FUNCTION = lossFunction

        self.costHistory, self.accuracyHistory = [], []
        self.accuracyTrained = self.testAccuracy = float('nan')
        self.costTrained = self.timeTrained = self.epochTrained = 0

        self.numEpochs = self.batchSize = 1
        self.epoch = self.batch = 0
        self.numBatches = None
        self.training = self.profiling = False
        self.trainDataBase = self.testDataBase = None

        self.NETWORK = self._constructNetwork()

    @abstractmethod
    def _constructNetwork(self) -> "Network":
        pass

    def process(self, _input) -> "np.ndarray":
        if self.training:
            warnings.showwarning("processing while training in progress may have unintended conflicts",
                                 ResourceWarning, 'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN
        return self.NETWORK.forwardPass(np.array(_input))

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
        assert isinstance(trainDataBase, DataBase)
        assert trainDataBase.inpShape == self.SHAPE.INPUT and trainDataBase.tarShape == self.SHAPE.OUTPUT
        if trainDataBase is not None or batchSize is not None:
            self.numBatches = int(np.ceil(self.trainDataBase.size / self.batchSize))

        if profile:
            self.profiling = True
            cProfile.run("self.train(test=test)")
            self.profiling = False
        else:
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
        assert isinstance(testDataBase, DataBase)
        assert testDataBase.inpShape == self.SHAPE.INPUT and testDataBase.tarShape == self.SHAPE.OUTPUT
        statPrinter('Testing', 'wait...', prefix=PrintCols.CBOLD + PrintCols.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            self.accuracyTrained = self.accuracy(self.trainDataBase.inputs, self.trainDataBase.targets)
        if testDataBase is not None: self.testAccuracy = self.accuracy(testDataBase.inputs, testDataBase.targets)
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
