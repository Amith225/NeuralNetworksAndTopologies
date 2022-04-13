import warnings as wr
from typing import *

import numpy as np
import numexpr as ne

from ..tools import NumpyDataCache, AbstractSave, AbstractLoad, Plot


class DataBase(AbstractSave, AbstractLoad):
    DEFAULT_DIR = '\\DataSets\\'
    DEFAULT_NAME = 'db'
    FILE_TYPE = '.zdb'
    LARGE_VAL = 5

    def saveName(self) -> str:
        return f"{self.size}s.{self.inpShape}i.{self.tarShape}o"

    def _write(self, dumpFile, *args, **kwargs):
        saveInputSet = self.inputSet * self.inputSetFactor
        if self.hotEncodeInp:
            saveInputSet = self.oneHotDecode(saveInputSet)
        saveTargetSet = self.targetSet * self.targetSetFactor
        if self.hotEncodeTar:
            saveTargetSet = self.oneHotDecode(saveTargetSet)
        np.savez_compressed(dumpFile,
                            inputSet=saveInputSet.astype(self.inputSetDtype),
                            targetSet=saveTargetSet.astype(self.targetSetDtype))

    @classmethod
    def _read(cls, loadFile, *args, **kwargs):
        nnLoader = np.load(loadFile, mmap_mode='r')
        try:
            inputSet, targetSet = nnLoader['arr_0'], nnLoader['arr_1']
        except KeyError:
            inputSet, targetSet = nnLoader['inputSet'], nnLoader['targetSet']

        return DataBase(inputSet, targetSet, *args, **kwargs)

    def __init__(self,
                 inputSet: Iterable and Sized,  # input signal
                 targetSet: Iterable and Sized,  # desired output signal
                 normalizeInp: float = None,
                 normalizeTar: float = None,
                 reshapeInp=None,
                 reshapeTar=None):
        if (size := len(inputSet)) != len(targetSet):
            raise Exception("Both input and output set should be of same size")
        self.inputSetDtype = inputSet.dtype
        self.targetSetDtype = targetSet.dtype
        self.hotEncodeInp = False
        self.hotEncodeTar = False
        if len(np.shape(inputSet)) == 1:
            inputSet = self.oneHotEncode(inputSet)
            self.hotEncodeInp = True
        if len(np.shape(targetSet)) == 1:
            targetSet = self.oneHotEncode(targetSet)
            self.hotEncodeTar = True
        if (maxI := np.max(inputSet)) >= self.LARGE_VAL and normalizeInp is None and not self.hotEncodeInp:
            wr.showwarning(f"inputSet has element(s) with values till {maxI} which may cause nan training, "
                           f"use of param 'normalizeInp=<max>' is recommended", FutureWarning, 'dataBase.py', 0)
        if (maxT := np.max(targetSet)) >= self.LARGE_VAL and normalizeTar is None and not self.hotEncodeTar:
            wr.showwarning(f"targetSet has element(s) with values till {maxT} which may cause nan training, "
                           f"use of param 'normalizeTar=<max>' is recommended", FutureWarning, 'dataBase.py', 0)

        inputSet, self.inputSetFactor = self.normalize(np.array(inputSet, dtype=np.float32), normalizeInp)
        targetSet, self.targetSetFactor = self.normalize(np.array(targetSet, dtype=np.float32), normalizeTar)
        if reshapeInp is not None:
            inputSet = inputSet.reshape((size, *reshapeInp))
        if reshapeTar is not None:
            inputSet = targetSet.reshape((size, *reshapeTar))
        self.inputSet = NumpyDataCache(inputSet)
        self.targetSet = NumpyDataCache(targetSet)

        self.size: int = size
        self.inpShape = inputSet.shape[1:]
        self.tarShape = targetSet.shape[1:]

        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

    @staticmethod
    def oneHotEncode(_1dArray):
        hotEncodedArray = np.zeros((len(_1dArray), max(_1dArray) + 1, 1))
        hotEncodedArray[np.arange(hotEncodedArray.shape[0]), _1dArray] = 1

        return hotEncodedArray

    @staticmethod
    def oneHotDecode(_3dArray):
        return np.where(_3dArray == 1)[1]

    # normalize input and target sets within the range of -scale to +scale
    @staticmethod
    def normalize(data, scale: float = None) -> Tuple["np.ndarray", float]:
        if scale is None:
            factor = 1
        else:
            factor = ne.evaluate("abs(data) * scale", local_dict={'data': data, 'scale': scale}).max()

        return data / factor, factor

    # shuffle the index order
    def randomize(self) -> "None":
        np.random.shuffle(self.indices)

    # returns a generator for input and target sets, each batch-sets of size batchSize at a time
    # send signal '-1' to end generator
    def batchGenerator(self, batch_size) -> Generator:
        if self.block:
            raise PermissionError("Access Denied: DataBase currently in use, "
                                  "end previous generator before creating a new one\n"
                                  "send signal '-1' to end generator or reach StopIteration")
        self.block = True
        self.batchSize = batch_size
        self.randomize()

        def generator() -> Generator:
            signal = yield
            while True:
                if signal == -1 or self.pointer + batch_size >= self.size:
                    rVal = self.__batch()
                    self.__resetVars()
                    yield rVal
                    return
                signal = yield self.__batch()
                self.pointer += batch_size
        gen = generator()
        gen.send(None)

        return gen

    # returns batch-set from index pointer to i
    def __batch(self) -> tuple[np.ndarray, np.ndarray]:
        vacant = 0
        if (i := self.pointer + self.batchSize) > self.size:
            i = self.size
            filled = i - self.pointer
            vacant = self.batchSize - filled
        indices = self.indices[self.pointer:i] + self.indices[:vacant]
        inputBatch = self.inputSet[indices]
        targetBatch = self.targetSet[indices]

        return inputBatch, targetBatch

    # resets generator flags after generator cycle
    def __resetVars(self):
        self.pointer = 0
        self.block = False
        self.batchSize = None


class PlotDataBase(Plot):
    @staticmethod
    def showMap():
        pass
