import sys
import time
from typing import TYPE_CHECKING

from .printCols import PrintCols

if TYPE_CHECKING:
    import numpy as np


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


# fixme: improve
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


def string_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load(name, raw_args, raw_kwargs, kwargs=None):
    if kwargs is None: kwargs = {}
    return string_import(name).__load__(raw_args, raw_kwargs, **kwargs)
