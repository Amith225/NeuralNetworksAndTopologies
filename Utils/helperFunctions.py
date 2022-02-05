import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..Topologies import AbstractActivationFunction
    from ..NeuralNetworks import _
import sys

import numpy as np


def copyNumpyList(lis: tp.List[np.ndarray]):
    copyList = []
    for array in lis:
        copyList.append(array.copy())

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
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
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
