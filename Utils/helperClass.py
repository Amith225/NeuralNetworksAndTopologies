import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..Topologies import AbstractActivationFunction
    from ..NeuralNetworks import _
import tempfile as tf

import numpy as np
from numpy.lib import format as fm

from .helperFunctions import iterable


class Shape:
    def __init__(self, *shape):
        self._shape = list(shape)
        self.__format_shape()
        self.LAYERS = len(self._shape)
        self.INPUT = self._shape[0]
        self.OUTPUT = self._shape[-1]

    def __getitem__(self, item):
        return self._shape[item]

    # todo: improve Shape.__format_shape
    def __format_shape(self):
        for i, layer in enumerate(self._shape):
            if not iterable(layer):
                self._shape[i] = (layer, 1)
        self._shape = tuple(self._shape)

    @property
    def shape(self):
        return self._shape


class Activators:
    def __init__(self, *activationFunctions: "AbstractActivationFunction"):
        self.activationFunctions = activationFunctions

    def __call__(self, length):
        return self.get(length)

    def get(self, length):
        activations = [None]
        activationDerivatives = [None]
        prevActivationFunction = None
        numEllipsis = self.activationFunctions.count(Ellipsis)
        numActivations = len(self.activationFunctions) - numEllipsis
        vacancy = length - numActivations
        for activationFunction in self.activationFunctions:
            if activationFunction == Ellipsis:
                for i in range(filled := (vacancy // numEllipsis)):
                    activations.append(prevActivationFunction.activation)
                    activationDerivatives.append(prevActivationFunction.activatedDerivative)
                vacancy -= filled
                numEllipsis -= 1
                continue
            prevActivationFunction = activationFunction
            activations.append(activationFunction.activation)
            activationDerivatives.append(activationFunction.activatedDerivative)

        return activations, activationDerivatives


class NumpyDataCache(np.ndarray):
    def __new__(cls, array):
        return cls.writeNpyCache(array)

    @staticmethod
    def writeNpyCache(array: "np.ndarray") -> np.ndarray:
        with tf.NamedTemporaryFile(suffix='.npy') as file:
            np.save(file, array)
            file.seek(0)
            fm.read_magic(file)
            fm.read_array_header_1_0(file)
            memMap = np.memmap(file, mode='r', shape=array.shape, dtype=array.dtype, offset=file.tell())

        return memMap
