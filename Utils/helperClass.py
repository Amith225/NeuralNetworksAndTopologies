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


class Collections:
    def __init__(self, *collectables):
        self.collectables = collectables

    def __call__(self, length):
        return self.get(length)

    def get(self, length):
        trueCollectables = []
        prevCollectable = None
        numEllipsis = self.collectables.count(Ellipsis)
        numCollectables = len(self.collectables) - numEllipsis
        vacancy = length - numCollectables
        for collectable in self.collectables:
            if collectable == Ellipsis:
                for i in range(filled := (vacancy // numEllipsis)):
                    trueCollectables.append(prevCollectable)
                vacancy -= filled
                numEllipsis -= 1
                continue
            trueCollectables.append(collectable)
            prevCollectable = collectable

        return trueCollectables


class Activators(Collections):
    def __init__(self, *activationFunctions: "AbstractActivationFunction"):
        super(Activators, self).__init__(*activationFunctions)

    def __call__(self, length):
        activations = [None]
        activationDerivatives = [None]
        for e in self.get(length):
            activations.append(e.activation)
            activationDerivatives.append(e.activatedDerivative)

        return activations, activationDerivatives


class Types(Collections):
    def __init__(self, *types):
        super(Types, self).__init__(*types)

    def __call__(self, length):
        return [None] + self.get(length)


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
