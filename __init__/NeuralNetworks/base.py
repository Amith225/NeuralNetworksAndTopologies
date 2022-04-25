from abc import ABCMeta, abstractmethod

import numpy as np

from ..tools import MetaPropertyGenerator, ReadOnlyProperty


class BaseShape(metaclass=MetaPropertyGenerator(ABCMeta)):
    def __str__(self):
        pass

    def __save__(self):
        pass

    def __init__(self, *shapes):
        assert len(shapes) > 0
        self.SHAPES = self._formatShapes(shapes)
        assert hash(self.SHAPES)
        self.LAYERS = len(self.SHAPES)
        self.INPUT = self.SHAPES[0]
        self.OUTPUT = self.SHAPES[-1]

    def __getitem__(self, item):
        return self.SHAPES[item]

    @staticmethod
    @abstractmethod
    def _formatShapes(shapes) -> tuple:
        """
        method to format given shapes
        :return: hashable formatted shapes
        """


class BaseLayer(metaclass=MetaPropertyGenerator(ABCMeta)):
    def __str__(self):
        pass

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape"):
        """
        :param shape: input, output, intermediate(optional) structure of the layer
        """
        self.SHAPE = shape

        self.input = ReadOnlyProperty(None)
        self.output = ReadOnlyProperty(None)
        self.givenDelta = ReadOnlyProperty(None)
        self.delta = ReadOnlyProperty(None)

    def forPass(self, _input: "np.ndarray"):
        """
        method for forward pass of inputs
        :param _input: self.output of the previous layer
        """
        self.input = _input
        self.output = self._fireAndFindOutput()

    def backProp(self, _delta: "np.ndarray"):
        """
        method for backward propagation of deltas
        :param _delta: self.delta of the following layer
        """
        self.givenDelta = _delta
        self.delta = self._wireAndFindDelta()

    @abstractmethod
    def _defineDeps(self, *depArgs, **depKwargs):
        """define all dependant objects ($deps) for the layer"""

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


class BaseNN(metaclass=MetaPropertyGenerator(ABCMeta)):
    def __str__(self):
        pass

    def __save__(self):
        pass
