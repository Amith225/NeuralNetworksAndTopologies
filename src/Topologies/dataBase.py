import warnings
from typing import Iterable, Sized, Generator

import numpy as np
import numexpr as ne

from ..tools import NumpyDataCache, BaseSave, BaseLoad, Plot


class DataBase(BaseSave, BaseLoad):
    """

    """
    DEFAULT_DIR = 'DataSets'
    DEFAULT_NAME = 'db'
    FILE_TYPE = '.zdb'
    LARGE_VAL = 5

    def __repr__(self):
        Shape = {'SIZE': self.size, 'BatchSize': self.batchSize, 'INPUT': self.inpShape, 'TARGET': self.tarShape}
        return f"<{self.__class__.__name__}:{self.NAME}:{Shape=}>"

    def __str__(self):
        HotEncode = {'INPUT': self.hotEncodeInp, 'TARGET': self.hotEncodeTar}
        SetDType = {'INPUT': self.inputSetDType, 'TARGET': self.targetSetDType}
        NormFactor = {'INPUT': f"Max:{(Max := self.inputMax)}, "
                               f"Norm:{(Norm := self.inputSetNormFactor)}, {Max*Norm=}",
                      'TARGET': f"Max:{(Max := self.targetMax)}, "
                                f"Norm:{(Norm := self.targetSetNormFactor)}, {Max*Norm=}"}
        return f"{self.__repr__()[:-1]}:\n\t{HotEncode=}\n\t{SetDType=}\n\t{NormFactor=}>"

    def saveName(self) -> str:
        return f"{self.size}s_{self.inpShape}i_{self.tarShape}o"

    def save(self, file: str = None, replace: bool = False) -> str:
        dumpFile = super(DataBase, self).save(file, replace)
        saveInputSet = self.inputs * self.inputSetNormFactor
        if self.hotEncodeInp: saveInputSet = self.oneHotDecode(saveInputSet)
        saveTargetSet = self.targets * self.targetSetNormFactor
        if self.hotEncodeTar: saveTargetSet = self.oneHotDecode(saveTargetSet)
        np.savez_compressed(dumpFile, inputSet=saveInputSet.astype(self.inputSetDType),
                            targetSet=saveTargetSet.astype(self.targetSetDType))
        return dumpFile

    @classmethod
    def load(cls, file: str, *DataBase_args, **DataBase_kwargs) -> "DataBase":
        f"""
        :param file: path like or name
        :param DataBase_args: to {DataBase.__init__}(normalizeInp, normalizeTar, reshapeInp, reshapeTar,
        oneHotMaxInp, oneHotMaxTar, name)
        """
        loadFile = super(DataBase, cls).load(file)
        nnLoader = np.load(loadFile, mmap_mode='r')
        inputSet, targetSet = nnLoader['inputSet'], nnLoader['targetSet']

        return DataBase(inputSet, targetSet, *DataBase_args, **DataBase_kwargs)

    def __getitem__(self, item):
        return self.inputs[(i := self.indices[item])], self.targets[i]

    def __init__(self,
                 inputSet: Iterable and Sized, targetSet: Iterable and Sized,
                 normalizeInp: float = None, normalizeTar: float = None,
                 reshapeInp=None, reshapeTar=None,
                 oneHotMaxInp=None, oneHotMaxTar=None,
                 name: str = ''):
        if (size := len(inputSet)) != len(targetSet): raise Exception("Both input and target set must be of same size")
        self.NAME = name
        self.inputSetDType, self.targetSetDType = inputSet.dtype, targetSet.dtype
        self.hotEncodeInp = self.hotEncodeTar = False
        if len(np.shape(inputSet)) == 1: inputSet, self.hotEncodeInp = self.oneHotEncode(inputSet, oneHotMaxInp)
        if len(np.shape(targetSet)) == 1: targetSet, self.hotEncodeTar = self.oneHotEncode(targetSet, oneHotMaxTar)
        if (maxI := np.max(inputSet)) >= self.LARGE_VAL and normalizeInp is None and not self.hotEncodeInp:
            warnings.showwarning(f"inputSet has element(s) with values till {maxI} which may cause nan training, "
                                 f"use of param 'normalizeInp=<max>' is recommended", FutureWarning, 'dataBase.py', 0)
        if (maxT := np.max(targetSet)) >= self.LARGE_VAL and normalizeTar is None and not self.hotEncodeTar:
            warnings.showwarning(f"targetSet has element(s) with values till {maxT} which may cause nan training, "
                                 f"use of param 'normalizeTar=<max>' is recommended", FutureWarning, 'dataBase.py', 0)

        inputSet, self.inputSetNormFactor = self.normalize(np.array(inputSet, dtype=np.float32), normalizeInp)
        targetSet, self.targetSetNormFactor = self.normalize(np.array(targetSet, dtype=np.float32), normalizeTar)
        self.inputMax, self.targetMax = inputSet.max(), targetSet.max()
        if reshapeInp is not None: inputSet = inputSet.reshape((size, *reshapeInp))
        if reshapeTar is not None: inputSet = targetSet.reshape((size, *reshapeTar))
        self.inputs, self.targets = NumpyDataCache(inputSet), NumpyDataCache(targetSet)

        self.size: int = size
        self.inpShape, self.tarShape = inputSet.shape[1:], targetSet.shape[1:]

        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

    @staticmethod
    def oneHotEncode(_1dArray, oneHotMax=None):
        if oneHotMax is None: oneHotMax = max(_1dArray) + 1
        hotEncodedArray = np.zeros((len(_1dArray), oneHotMax, 1))
        hotEncodedArray[np.arange(hotEncodedArray.shape[0]), _1dArray] = 1

        return hotEncodedArray, oneHotMax

    @staticmethod
    def oneHotDecode(_3dArray):
        return np.where(_3dArray == 1)[1]

    # normalize input and target sets within the range of -scale to +scale
    @staticmethod
    def normalize(data, scale: float = None) -> tuple["np.ndarray", float]:
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
    def batchGenerator(self, batchSize) -> Generator[tuple["np.ndarray", "np.ndarray"], None, None]:
        if self.block:
            raise PermissionError("Access Denied: DataBase currently in use, "
                                  "end previous generator before creating a new one\n"
                                  "send signal '-1' to end generator or reach StopIteration")
        self.block = True
        self.batchSize = batchSize
        self.randomize()

        def generator() -> Generator:
            signal = yield
            while True:
                if signal == -1 or self.pointer + batchSize >= self.size:
                    rVal = self.__batch()
                    self.__resetVars()
                    yield rVal
                    return
                signal = yield self.__batch()
                self.pointer += batchSize

        gen = generator()
        gen.send(None)
        return gen

    def __batch(self) -> tuple[np.ndarray, np.ndarray]:
        indices = self.indices[self.pointer:self.pointer + self.batchSize]
        inputBatch = self.inputs[indices]
        targetBatch = self.targets[indices]

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
