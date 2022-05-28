import tempfile
import ctypes

import numpy as np
from numpy.lib import format as fm


# fixme: just make as function
class NumpyDataCache(np.ndarray):
    def __new__(cls, array):
        return cls.writeNpyCache(array)

    @staticmethod
    def writeNpyCache(array: "np.ndarray") -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix='.npy') as file:
            np.save(file, array)
            file.seek(0)
            fm.read_magic(file)
            fm.read_array_header_1_0(file)
            memMap = np.memmap(file, mode='r', shape=array.shape, dtype=array.dtype, offset=file.tell())

        return memMap


class NewCopy:
    __args, __kwargs = (), {}

    def __new__(cls, *args, **kwargs):
        cls.__args, cls.__kwargs = args, kwargs
        obj = super(NewCopy, cls).__new__(cls)
        obj.__init__(*args, **kwargs)  # noqa
        return obj

    @classmethod
    def __new_copy__(cls):
        return cls.__new__(cls, *cls.__args, *cls.__kwargs)


class Collections:
    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.collectables}>"

    # todo::
    def __save__(self):
        raise NotImplementedError

    def __load__(self):
        raise NotImplementedError

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
                filled = vacancy // numEllipsis
                trueCollectables.extend([prevCollectable] * filled)
                vacancy -= filled
                numEllipsis -= 1
                continue
            trueCollectables.append(collectable)
            prevCollectable = collectable

        return trueCollectables


try:
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except AttributeError:
    pass
# todo: use NamedTuple
class PrintCols:  # noqa
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBOLDITALIC = CBOLD + CITALIC
    CURLBOLD = CBOLD + CURL
    CITALICURL = CITALIC + CURL
    CBOLDITALICURL = CBOLD + CITALIC + CURL

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'
