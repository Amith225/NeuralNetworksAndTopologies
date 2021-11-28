import os as _os
import typing as _tp

import numpy as _np
import numexpr as _ne

from utils import NumpyDataCache

if _tp.TYPE_CHECKING:
    pass


class DataBase:
    def __init__(self, inputSet: _tp.Iterable and _tp.Sized,  # input signal
                 targetSet: _tp.Iterable and _tp.Sized,  # desired output signal
                 normalize: _tp.Union[int, float] = 0):
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
    def normalize(data, scale: _tp.Union[int, float] = 1) -> _tp.Tuple["_np.ndarray", float]:
        if scale != 0:  # do not normalize if scale is zero
            factor = _ne.evaluate("abs(data) * scale", local_dict={'data': data, 'scale': scale}).max()
        else:
            factor = 1

        return data / factor, factor

    # load a database file
    @staticmethod
    def load(file: str, normalize: _tp.Union[int, float] = 0) -> "DataBase":
        if file:
            if not _os.path.dirname(file):
                file = _os.getcwd() + '\\DataSets\\' + file
        else:
            raise FileExistsError("file not given")

        if file[-4:] != '.npz':
            raise ValueError(f"file type must be that of 'NPZ' but given {file}")
        nnLoader = _np.load(file)

        return DataBase(nnLoader['arr_0'], nnLoader['arr_1'], normalize)

    # save database as NPZ file
    def save(self, fname: str = None, dtype=_np.int8) -> str:
        if fname is None:
            fname = 'db'
        fpath = _os.getcwd() + '\\DataSets\\'
        spath = fpath + fname + f's{self.size}i{self.inpShape}o{self.tarShape}'

        i = 0
        nSpath = spath
        while 1:
            if i != 0:
                nSpath = spath + ' (' + str(i) + ')'
            if _os.path.exists(nSpath + '.nndb' + '.npz'):
                i += 1
            else:
                spath = nSpath
                break
        _os.makedirs(fpath, exist_ok=True)
        _np.savez_compressed(spath + '.nndb',
                             (self.inputSet[:] * self.inputSetFactor).astype(dtype),
                             (self.targetSet[:] * self.targetSetFactor).astype(dtype))

        return spath

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
