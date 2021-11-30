import typing as tp
from abc import ABCMeta, abstractmethod
import os

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


class AbstractSave(metaclass=ABCMeta):
    DEFAULT_DIR: str
    DEFAULT_NAME: str
    FILE_TYPE: str

    @abstractmethod
    def saveName(self) -> str:
        pass

    @abstractmethod
    def _write(self, dumpFile, *args, **kwargs):
        pass

    def save(self, file: str = None, replace: bool = False, *args, **kwargs) -> str:
        if file is None:
            file = self.DEFAULT_NAME
        if not (fpath := os.path.dirname(file)):
            fpath = os.getcwd() + self.DEFAULT_DIR
            fname = file
        else:
            fpath += '\\'
            fname = os.path.basename(file)
        os.makedirs(fpath, exist_ok=True)
        if len(fname) >= 1 + len(self.FILE_TYPE) and fname[-4:] == self.FILE_TYPE:
            fname = fname[:-4]
        savePath = fpath + fname + self.saveName()

        i = 0
        numSavePath = savePath
        if not replace:
            while 1:
                if i != 0:
                    numSavePath = savePath + ' (' + str(i) + ')'
                if os.path.exists(numSavePath + self.FILE_TYPE):
                    i += 1
                else:
                    break

        with open(finalPath := (numSavePath + self.FILE_TYPE), 'wb') as dumpFile:
            self._write(dumpFile, *args, **kwargs)

        return finalPath


class AbstractLoad(metaclass=ABCMeta):
    DEFAULT_DIR: str
    FILE_TYPE: str

    @classmethod
    @abstractmethod
    def _read(cls, loadFile, *args, **kwargs):
        pass

    @classmethod
    def load(cls, file: str, *args, **kwargs):
        if file:
            if not (fpath := os.path.dirname(file)):
                fpath = os.getcwd() + cls.DEFAULT_DIR
                fname = file
            else:
                fpath += '\\'
                fname = os.path.basename(file)
        else:
            raise NameError("file not given")
        if '.' not in fname:
            fname += cls.FILE_TYPE

        with open(fpath + fname, 'rb') as loadFile:
            rVal = cls._read(loadFile, *args, **kwargs)

        return rVal


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
