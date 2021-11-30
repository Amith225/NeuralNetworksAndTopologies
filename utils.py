import typing as tp

import numpy as np
from numpy.lib import format as fm
import tempfile as tf

if tp.TYPE_CHECKING:
    from Topologies.activationFuntion import AbstractActivationFunction


class WBShape:
    def __init__(self, *wbShape):
        self._shape = tuple(wbShape)
        self.LAYERS = len(self._shape)

    def __getitem__(self, item):
        return self._shape[item]

    def shape(self):
        return self._shape


class Activators:
    def __init__(self, *activationFunctions: "AbstractActivationFunction"):
        self.activationFunctions = activationFunctions

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


class NumpyDataCache:
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


def copyNumpyList(lis: tp.List[np.ndarray]):
    copyList = []
    for array in lis:
        copyList.append(array.copy())

    return copyList


def iterable(var):
    try:
        iter(var)
        return True
    except TypeError:
        return False
