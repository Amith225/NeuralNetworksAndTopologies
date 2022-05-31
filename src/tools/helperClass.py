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


class Dunder(tuple):
    __slots__ = {}


class DunderSaveLoad:
    __RAW_ARGS, __RAW_KWARGS = (), {}
    _dict = False

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self.__RAW_ARGS = args
        self.__RAW_KWARGS = kwargs
        return self

    def __save__(self):
        cls_name = f"{self.__module__}.{type(self).__name__}"
        return cls_name, self.checkForDunderObjects(self.__RAW_ARGS, "encode"), \
               self.checkForDunderObjects(self.__RAW_KWARGS, "encode"), \
               *([] if not self._dict else [{'_dict': self.checkForDunderObjects(self.__dict__, "encode")}])

    @classmethod
    def __load__(cls, raw_args, raw_kwargs, **kwargs):
        raw_args = cls.checkForDunderObjects(raw_args, "decode")
        raw_kwargs = cls.checkForDunderObjects(raw_kwargs, "decode")
        self = cls(*raw_args, **raw_kwargs)
        if self._dict and '_dict' in kwargs: self.__dict__.update(self.checkForDunderObjects(kwargs['_dict'], "decode"))
        return self

    @classmethod
    def checkForDunderObjects(cls, _obj, _type):
        assert _type in (types := ("encode", "decode"))
        if isinstance(_obj, dict):
            keys, vals = _obj.keys(), _obj.values()
            return {key: item for item, key in zip(cls.checkForDunderObjects(list(vals), _type), keys)}
        elif isinstance(_obj, list):
            return [cls.checkForDunderObjects(ob, _type) for ob in _obj]
        elif isinstance(_obj, tuple) and not isinstance(_obj, Dunder):
            return tuple([cls.checkForDunderObjects(ob, _type) for ob in _obj])
        else:
            if _type == types[0] and isinstance(_obj, DunderSaveLoad):
                return Dunder(_obj.__save__())
            elif _type == types[1] and isinstance(_obj, Dunder):
                return load(*_obj)
            else:
                return _obj
