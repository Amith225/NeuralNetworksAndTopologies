import os as _os
import typing as _tp
import tempfile as _tf

import numpy as _np
import numexpr as _ne
from numpy.lib import format as _format

from utils import iterable, fancyMax, fancyMin

if _tp.TYPE_CHECKING:
    pass


class DataCache:
    def __init__(self, data):
        self.__cache = self.writeNpyCache(data)
        self.__fHandle = open(self.__cache.name, 'rb')
        self.__fHandle.seek(0)
        _ = _format.read_magic(self.__fHandle)
        self.shape, _, self.dtype = _format.read_array_header_1_0(self.__fHandle)
        self.fluke = self.__fHandle.tell()
        self.rowSize = int(_np.prod(self.shape[1:]))

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if step is None:
                step = 1
            return self.readNpyCache(range(start, stop, step))
        elif isinstance(item, int):
            return self.readNpyCache([item])[0]
        elif iterable(item):
            return self.readNpyCache(item)

    @property
    def cache(self):
        return self.__cache

    @staticmethod
    def writeNpyCache(data):
        file = _tf.NamedTemporaryFile()
        fname = file.name + '.npy'
        _np.save(fname, data)
        file.name += '.npy'

        return file

    def readNpyCache(self, indices):
        buffer = []
        for i, index in enumerate(indices):
            self.__fHandle.seek(index * self.rowSize * self.dtype.itemsize + self.fluke)
            # buf = _np.empty(self.shape[1:])
            # self.__fHandle.readinto(buf)
            # buffer.append(buf)
            buffer.append(_np.fromfile(self.__fHandle, count=self.rowSize, dtype=self.dtype))
        buffer = _np.array(buffer).reshape((-1, *self.shape[1:]))

        return buffer


class FutureDataBase:
    def __init__(self, inputSet: _tp.Iterable and _tp.Sized,  # input signal
                 targetSet: _tp.Iterable and _tp.Sized,  # desired output signal
                 normalize: _tp.Union[int, float] = 0):
        if (size := len(inputSet)) != len(targetSet):
            raise Exception("Both input and output set should be of same size")

        inputSet, self.inputSetFactor = self.normalize(_np.array(inputSet, dtype=_np.float32), normalize)
        targetSet, self.targetSetFactor = self.normalize(_np.array(targetSet, dtype=_np.float32), normalize)
        self.inputSetCache = DataCache(inputSet)
        self.targetSetCache = DataCache(targetSet)

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
    def load(file: str, normalize: _tp.Union[int, float] = 0) -> "FutureDataBase":
        if file:
            if not _os.path.dirname(file):
                file = _os.getcwd() + '\\DataSets\\' + file
        else:
            raise FileExistsError("file not given")

        if file[-4:] != '.npz':
            raise ValueError(f"file type must be that of 'NPZ' but given {file}")
        nnLoader = _np.load(file)

        return FutureDataBase(nnLoader['arr_0'], nnLoader['arr_1'], normalize)

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
                             (self.inputSetCache[:] * self.inputSetFactor).astype(dtype),
                             (self.targetSetCache[:] * self.targetSetFactor).astype(dtype))

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
        # self.randomize()

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
        inputBatch = self.inputSetCache[indices]
        targetBatch = self.targetSetCache[indices]

        return inputBatch, targetBatch

    # resets generator flags after generator cycle
    def __resetVars(self):
        self.pointer = 0
        self.block = False
        self.batchSize = None


class DataBase:
    def __init__(self, inputSet: _tp.Iterable and _tp.Sized,  # input signal
                 targetSet: _tp.Iterable and _tp.Sized,  # desired output signal
                 normalize: _tp.Union[int, float] = 0):
        if (size := len(inputSet)) != len(targetSet):
            raise Exception("Both input and output set should be of same size")

        self.inputSet = _np.array(inputSet, dtype=_np.float32)
        self.targetSet = _np.array(targetSet, dtype=_np.float32)

        self.size: int = size
        self.inpShape = self.inputSet.shape[1]
        self.tarShape = self.targetSet.shape[1]
        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

        self.normalize(normalize)

    # save database as NPZ file
    def save(self, fname: str = None) -> str:
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
        _np.savez_compressed(spath + '.nndb', self.inputSet, self.targetSet)

        return spath

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

    # normalize input and target sets within the range of -scale to +scale
    def normalize(self, scale: _tp.Union[int, float] = 1) -> "None":
        if scale != 0:  # do not normalize if scale is zero
            inputScale = _ne.evaluate("abs(inputSet) * scale",
                                      local_dict={'inputSet': self.inputSet, 'scale': scale}).max()
            targetScale = _ne.evaluate("abs(targetSet) * scale",
                                       local_dict={'targetSet': self.targetSet, 'scale': scale}).max()
            self.inputSet /= inputScale
            self.targetSet /= targetScale

    # shuffle the input and target sets randomly and simultaneously
    def randomize(self) -> "None":
        _np.random.shuffle(self.indices)
        self.inputSet = self.inputSet[self.indices]
        self.targetSet = self.targetSet[self.indices]

    # returns a generator for input and target sets, each batch-sets of size batchSize at a time
    def batchGenerator(self, batch_size) -> _tp.Generator:
        if self.block:
            raise PermissionError("Access Denied: DataBase currently in use, "
                                  "end previous generator before creating a new one")
        self.block = True
        self.batchSize = batch_size
        # self.randomize()

        def generator() -> _tp.Generator:
            while True:
                if (i := self.pointer + batch_size) >= self.size:
                    i = self.size
                    rVal = self.__batch(i)
                    self.__resetVars()
                    yield rVal
                    return
                signal = yield self.__batch(i)
                if signal == -1:
                    return self.__resetVars()
                self.pointer += batch_size

        return generator()

    # returns batch-set of index i
    def __batch(self, i) -> _tp.Tuple[_np.ndarray, _np.ndarray]:
        inputBatch = self.inputSet[self.pointer:i]
        targetBatch = self.targetSet[self.pointer:i]
        if (filled := i - self.pointer) != self.batchSize:
            vacant = self.batchSize - filled
            inputBatch = _np.append(inputBatch, self.inputSet[:vacant]).reshape([-1, *self.inputSet.shape[1:]])
            targetBatch = _np.append(targetBatch, self.targetSet[:vacant]).reshape([-1, *self.targetSet.shape[1:]])

        return inputBatch, targetBatch

    # resets generator flags after generator cycle
    def __resetVars(self):
        self.pointer = 0
        self.block = False
        self.batchSize = None
