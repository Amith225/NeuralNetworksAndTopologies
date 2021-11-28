import typing as tp

import numpy as np
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
    def __init__(self, data):
        file = self.writeNpyCache(data)
        self.__npLoader = np.load(file.name, mmap_mode='r')

    def __getitem__(self, item):
        return self.__npLoader[item]

    @staticmethod
    def writeNpyCache(data):
        file = tf.NamedTemporaryFile()
        fname = file.name + '.npy'
        np.save(fname, data)
        file.name += '.npy'

        return file


def copyNumpyList(lis: tp.List[np.ndarray]):
    copyList = []
    for array in lis:
        copyList.append(array.copy())

    return copyList
