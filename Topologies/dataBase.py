# library direct imports
import os
from typing import *

# library imports
import numpy as np
import numexpr as ne


class DataBase:  # main class
    def __init__(self, inputSet: "Iterable" and "Sized",  # input signal
                 targetSet: "Iterable" and "Sized",  # desired output signal
                 normalize: Union[int, float] = 0):  #
        if (size := len(inputSet)) != len(targetSet):
            raise Exception("Both input and output set should be of same size")

        self.inputSet = np.array(inputSet, dtype=np.float32)
        self.targetSet = np.array(targetSet, dtype=np.float32)

        self.size: int = size
        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

        self.normalize(normalize)

    # save database as NPZ file
    def save(self, fname: str = None) -> str:
        if fname is None:
            fname = 'db'
        fpath = os.getcwd() + '\\DataSets\\'
        spath = fpath + fname + f's{self.size}i{self.inputSet.shape[1]}o{self.targetSet.shape[1]}'

        i = 0
        nSpath = spath
        while 1:
            if i != 0:
                nSpath = spath + ' (' + str(i) + ')'
            if os.path.exists(nSpath + '.nndb' + '.npz'):
                i += 1
            else:
                spath = nSpath
                break
        os.makedirs(fpath, exist_ok=True)
        np.savez_compressed(spath + '.nndb', self.inputSet, self.targetSet)

        return spath

    # load a database file
    @staticmethod
    def load(file: str) -> "DataBase":
        if file:
            if not os.path.dirname(file):
                file = os.getcwd() + '\\DataSets\\' + file
        else:
            raise FileExistsError("file not given")

        if file[-4:-1] != '.npz':
            raise ValueError("file type must be that of 'NPZ'")
        nnLoader = np.load(file)

        return DataBase(nnLoader['arr_0'], nnLoader['arr_1'])

    # normalize input and target sets within the range of -scale to +scale
    def normalize(self, scale: Union[int, float] = 1) -> "None":
        if scale != 0:  # do not normalize if scale is zero
            inputScale = ne.evaluate("abs(inputSet)", local_dict={'inputSet': self.inputSet}).max()
            targetScale = ne.evaluate("abs(targetSet)", local_dict={'targetSet': self.targetSet}).max()
            self.inputSet /= inputScale * scale
            self.targetSet /= targetScale * scale

    # shuffle the input and target sets randomly and simultaneously
    def randomize(self) -> "None":
        np.random.shuffle(self.indices)
        self.inputSet = self.inputSet[self.indices]
        self.targetSet = self.targetSet[self.indices]

    # returns a generator for input and target sets, each batch-sets of size batchSize at a time
    def batchGenerator(self, batch_size) -> "Generator":
        if self.block:
            raise PermissionError("Access Denied: DataBase currently in use, "
                                  "'end' previous generator before creating a new one")
        self.block = True
        self.batchSize = batch_size
        self.randomize()

        def generator() -> "Generator":
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
    def __batch(self, i) -> Tuple[np.ndarray, np.ndarray]:
        inputBatch = self.inputSet[self.pointer:i]
        targetBatch = self.targetSet[self.pointer:i]
        if (filled := i - self.pointer) != self.batchSize:
            vacant = self.batchSize - filled
            inputBatch = np.append(inputBatch, self.inputSet[:vacant]).reshape([-1, *self.inputSet.shape[1:]])
            targetBatch = np.append(targetBatch, self.targetSet[:vacant]).reshape([-1, *self.targetSet.shape[1:]])

        return inputBatch, targetBatch

    # resets generator flags after generator cycle
    def __resetVars(self):
        self.pointer = 0
        self.block = False
        self.batchSize = None
