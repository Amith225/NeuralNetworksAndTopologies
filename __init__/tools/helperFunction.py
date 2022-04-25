import sys
import inspect

import numpy as np


def copyNumpyList(lis: list[np.ndarray]):
    copyList = []
    for array in lis: copyList.append(array.copy())

    return copyList


def iterable(var):
    try:
        iter(var)
        return True
    except TypeError:
        return False


def getSize(obj, seen=None, ref=''):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None: seen = set()
    if (obj_id := id(obj)) in seen: return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    ref += str(obj.__class__)
    if isinstance(obj, dict):
        size += sum([getSize(obj[k], seen, ref + str(k)) for k in obj.keys()])
        size += sum([getSize(k, seen, ref) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += getSize(obj.__dict__, seen, ref)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([getSize(i, seen, ref) for i in obj])

    if size > 1024 * 10:  # show files > 10Mb
        print(obj.__class__, size)
        print(ref, '\n')

    return size


class ReadOnlyProperty:
    def __init__(self, val, constant=False):
        self.__val = val
        self.__constant = constant

        self.__obj = (caller := inspect.stack()[1][0].f_locals).get('self',
                                                                    caller.get('cls', caller.get('__qualname__')))

    def __getitem__(self, item):
        return self.__val[item]

    def __getattr__(self, item):
        return self.__val.__getattr__(item)

    def __get__(self, instance, owner):
        return self.__val

    def __setitem__(self, key, value):
        if self.__magic__():
            self.__val[key] = value
            return
        raise AttributeError("attribute is read only")

    def __setattr__(self, key, value):
        if self.__magic__():
            self.__val.__setattr__(key, value)
            return
        raise AttributeError("attribute is read only")

    def __set__(self, instance, value):
        if self.__magic__():
            self.__val = value
            return
        raise AttributeError("attribute is read only")

    def __magic__(self):
        return not self.__constant and (
                (cls := inspect.stack()[1][0].f_locals.get('self').__class__) is self.__obj or any(
                 base.__name__ == self.__obj for base in cls.__bases__))


def MetaPropertyGenerator(*inherits):
    class MetaProperty(*inherits, type):
        def __call__(cls, *args, **kwargs):
            __obj = super(MetaProperty, cls).__call__(*args, **kwargs)
            magicProperty = set()
            if hasattr(__obj, "__magic_init__"):
                keyOld = set(__obj.__dict__.keys())
                __obj.__magic_init__()
                keyNew = set(__obj.__dict__.keys())
                magicProperty = keyNew - keyOld
            for key, val in __obj.__dict__.items():
                if key.isupper() and not (len(key) >= 2 and key[:2] == '__'):
                    setattr(cls, key, property(lambda self, __val=val: __val))
                elif key in magicProperty:
                    setattr(cls, key, ReadOnlyProperty(val))
            return __obj

    return MetaProperty
