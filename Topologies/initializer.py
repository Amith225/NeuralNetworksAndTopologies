import typing as _tp
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np

# library imports for type checking
if _tp.TYPE_CHECKING:
    from . import *
    from NeuralNetworks import *

# setup list or element numpy array of None
_np.NONE = [_np.array([None])]


class WBInitializer(metaclass=_ABCMeta):  # main class
    @_abstractmethod
    def __init__(self, *args, **kwargs):
        self.sn = _np.random.default_rng().standard_normal

    @_abstractmethod
    def initialize(self, shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
        pass


class UniformWBInitializer(WBInitializer):
    def __init__(self, start: _tp.Union["int", "float"] = -1, stop: _tp.Union["int", "float"] = 1):
        super(UniformWBInitializer, self).__init__()
        self.start = start
        self.stop = stop

    def initialize(self, shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
        biases = [_np.random.uniform(self.start, self.stop, (shape[i], 1)).astype(dtype=_np.float32)
                  for i in range(1, shape.LAYERS)]
        weights = [_np.random.uniform(self.start, self.stop, (shape[i], shape[i - 1])).astype(dtype=_np.float32)
                   for i in range(1, shape.LAYERS)]

        return _np.NONE + biases, _np.NONE + weights


class NormalWBInitializer(WBInitializer):
    def __init__(self, scale: _tp.Union["int", "float"] = 1):
        super(NormalWBInitializer, self).__init__()
        self.scale = scale

    def initialize(self, shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
        biases = [(self.sn((shape[i], 1), dtype=_np.float32)) * self.scale for i in range(1, shape.LAYERS)]
        weights = [(self.sn((shape[i], shape[i - 1]), dtype=_np.float32)) * self.scale for i in range(1, shape.LAYERS)]

        return _np.NONE + biases, _np.NONE + weights


class XavierWBInitializer(WBInitializer):
    def __init__(self, he: _tp.Union["int", "float"] = 1):
        super(XavierWBInitializer, self).__init__()
        self.he = he

    def initialize(self, shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
        biases = [self.sn((shape[i], 1), dtype=_np.float32) * (self.he / shape[i - 1]) ** 0.5
                  for i in range(1, shape.LAYERS)]
        weights = [self.sn((shape[i], shape[i - 1]), dtype=_np.float32) * (self.he / shape[i - 1]) ** 0.5
                   for i in range(1, shape.LAYERS)]

        return _np.NONE + biases, _np.NONE + weights


class NormalizedXavierWBInitializer(WBInitializer):
    def __init__(self, he: _tp.Union["int", "float"] = 6):
        super(NormalizedXavierWBInitializer, self).__init__()
        self.he = he

    def initialize(self, shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
        biases = [self.sn((shape[i], 1), dtype=_np.float32) * (self.he / (shape[i - 1] + shape[i])) ** 0.5
                  for i in range(1, shape.LAYERS)]
        weights = [self.sn((shape[i], shape[i - 1]), dtype=_np.float32) * (self.he / (shape[i - 1] + shape[i])) ** 0.5
                   for i in range(1, shape.LAYERS)]

        return _np.NONE + biases, _np.NONE + weights
