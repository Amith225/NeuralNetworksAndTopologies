import tempfile

import numpy as np
from numpy.lib import format as fm

from .helperFunction import load, iterable


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


class Collections:
    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.collectables}>"

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


class Dunder:
    def __init__(self, save):
        self.save = save


class DunderSaveLoad:
    __RAW_ARGS, __RAW_KWARGS = (), {}
    _dict = False

    def __new__(cls, *args, **kwargs):
        cls.__RAW_ARGS = [arg if not isinstance(arg, DunderSaveLoad) else Dunder(arg.__save__())
                          for arg in args]
        cls.__RAW_KWARGS = {key: arg if not isinstance(arg, DunderSaveLoad) else Dunder(arg.__save__())
                            for key, arg in kwargs.items()}
        return super().__new__(cls)

    def __save__(self):
        cls_name = f"{self.__module__}.{type(self).__name__}"
        return cls_name, self.__RAW_ARGS, self.__RAW_KWARGS, *([] if not self._dict else [{'_dict': self.__dict__}])

    @classmethod
    def __load__(cls, raw_args, raw_kwargs, **kwargs):
        raw_args = [load(*arg.save) if isinstance(arg, Dunder) else arg for arg in raw_args]
        raw_kwargs = {key: load(*arg.save) if isinstance(arg, Dunder) else arg for key, arg in raw_kwargs.items()}
        self = cls(*raw_args, **raw_kwargs)
        if self._dict: self.__dict__.update(kwargs['_dict'])
        return self

    # todo:
    def checkForDunderObjects(self, _obj):
        raise NotImplementedError
