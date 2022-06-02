import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from .printCols import PrintCols


def copyNumpyList(lis: list["np.ndarray"]):
    copyList = []
    for array in lis: copyList.append(array.copy())
    return copyList


def iterable(var):
    try:
        iter(var)
        return True
    except TypeError:
        return False


def secToHMS(seconds, hms=('h', 'm', 's')):
    encode = f'%S{hms[2]}'
    if (tim := time.gmtime(seconds)).tm_min != 0: encode = f'%M{hms[1]}' + encode
    if tim.tm_hour != 0: encode = f'%H{hms[0]}' + encode
    return time.strftime(encode, tim)


def statPrinter(key, value, *, prefix='', suffix=PrintCols.CEND, end=' '):
    print(prefix + f"{key}:{value}" + suffix, end=end)


def getSize(obj, seen=None, thresh=1024):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None: seen = set()
    if (obj_id := id(obj)) in seen: return 0, {}
    seen.add(obj_id)
    tree = {}
    if isinstance(obj, dict):
        for k in obj.keys():
            siz1, tre, siz2, tre2 = *getSize(obj[k], seen, thresh), *getSize(k, seen, thresh)  # noqa
            tre.update(tre2)
            siz = siz1 + siz2
            if siz >= thresh: tree[k] = siz, tre
            size += siz
    elif hasattr(obj, '__dict__'):
        siz, tree = getSize(obj.__dict__, seen, thresh)
        size += siz
    elif isinstance(obj, np.ndarray):
        size = obj.nbytes
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for j, i in enumerate(obj):
            siz, tre = getSize(i, seen, thresh)
            if siz >= thresh: tree[j] = siz, tre
            size += siz
    return size, tree


def string_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load(name, raw_args, raw_kwargs, kwargs=None):
    if kwargs is None: kwargs = {}
    return string_import(name).__load__(raw_args, raw_kwargs, **kwargs)
