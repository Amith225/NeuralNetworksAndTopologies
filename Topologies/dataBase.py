import typing as _tp

import numpy as _np
import numexpr as _ne

from utils import NumpyDataCache, AbstractSave, AbstractLoad

if _tp.TYPE_CHECKING:
    pass


class DataBase(AbstractSave, AbstractLoad):
    DEFAULT_DIR = '\\DataSets\\'
    DEFAULT_NAME = 'db'
    FILE_TYPE = '.zdb'

    def saveName(self) -> str:
        return f"{self.size}s{self.inpShape}i{self.tarShape}o"

    # make dtype as the type which was imported ##############
    def _write(self, dumpFile, *args, **kwargs):
        dtype = _np.uint8
        if 'dtype' in kwargs.keys():
            dtype = kwargs['dtype']
        _np.savez_compressed(dumpFile,
                             inputSet=(self.inputSet * self.inputSetFactor).astype(dtype),
                             targetSet=(self.targetSet * self.targetSetFactor).astype(dtype))

    @classmethod
    def _read(cls, loadFile, *args, **kwargs):
        nnLoader = _np.load(loadFile, mmap_mode='r')
        try:
            inputSet, targetSet = nnLoader['arr_0'], nnLoader['arr_1']
        except KeyError:
            inputSet, targetSet = nnLoader['inputSet'], nnLoader['targetSet']

        return DataBase(inputSet, targetSet, *args, **kwargs)

    def __init__(self, inputSet: _tp.Iterable and _tp.Sized,  # input signal
                 targetSet: _tp.Iterable and _tp.Sized,  # desired output signal
                 normalize: float = None):
        if (size := len(inputSet)) != len(targetSet):
            raise Exception("Both input and output set should be of same size")

        inputSet, self.inputSetFactor = self.normalize(_np.array(inputSet, dtype=_np.float32), normalize)
        targetSet, self.targetSetFactor = self.normalize(_np.array(targetSet, dtype=_np.float32), normalize)
        self.inputSet = NumpyDataCache(inputSet)
        self.targetSet = NumpyDataCache(targetSet)

        self.size: int = size
        self.inpShape = inputSet.shape[1]
        self.tarShape = targetSet.shape[1]
        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

    # normalize input and target sets within the range of -scale to +scale
    @staticmethod
    def normalize(data, scale: float = None) -> _tp.Tuple["_np.ndarray", float]:
        if scale is None:
            factor = 1
        else:
            factor = _ne.evaluate("abs(data) * scale", local_dict={'data': data, 'scale': scale}).max()

        return data / factor, factor

    # shuffle the index order
    def randomize(self) -> "None":
        _np.random.shuffle(self.indices)

    # returns a generator for input and target sets, each batch-sets of size batchSize at a time
    # send signal '-1' to end generator
    def batchGenerator(self, batch_size) -> _tp.Generator:
        if self.block:
            raise PermissionError("Access Denied: DataBase currently in use, "
                                  "end previous generator before creating a new one\n"
                                  "send signal '-1' to end generator or reach StopIteration")
        self.block = True
        self.batchSize = batch_size
        self.randomize()

        def generator() -> _tp.Generator:
            signal = None
            while True:
                if signal == -1 or self.pointer + batch_size >= self.size:
                    rVal = self.__batch()
                    self.__resetVars()
                    yield rVal
                    return
                signal = yield self.__batch()
                self.pointer += batch_size

        return generator()

    # returns batch-set from index pointer to i
    def __batch(self) -> _tp.Tuple[_np.ndarray, _np.ndarray]:
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
