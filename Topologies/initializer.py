import typing as tp
if tp.TYPE_CHECKING:
    from . import *
    from ..NeuralNetworks import *
    from ..Utils import *

import numpy as np

from abc import ABCMeta, abstractmethod

# setup list or element numpy array of None
np.NONE = [np.array([None])]


class WBInitializer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.sn = np.random.default_rng().standard_normal

    def __call__(self, uniShape):
        return self._initialize(uniShape)

    @abstractmethod
    def _initialize(self, uniShape):
        pass


# todo: take single shape and return values
class UniformWBInitializer(WBInitializer):
    def __init__(self, start: tp.Union["int", "float"] = -1, stop: tp.Union["int", "float"] = 1):
        super(UniformWBInitializer, self).__init__()
        self.start = start
        self.stop = stop

    def _initialize(self, uniShape):
        return np.NONE + [np.random.uniform(self.start, self.stop, shape).astype(dtype=np.float32)
                          for shape in uniShape[1:]]


class NormalWBInitializer(WBInitializer):
    def __init__(self, scale: tp.Union["int", "float"] = 1):
        super(NormalWBInitializer, self).__init__()
        self.scale = scale

    def _initialize(self, uniShape):
        return np.NONE + [self.sn(shape, dtype=np.float32) * self.scale for shape in uniShape[1:]]


class XavierWBInitializer(WBInitializer):
    def __init__(self, he: tp.Union["int", "float"] = 1):
        super(XavierWBInitializer, self).__init__()
        self.he = he

    def _initialize(self, uniShape):
        np.NONE + [self.sn(shape, dtype=np.float32) * (self.he / np.prod(uniShape[i - 1])**len(uniShape[i - 1])) ** 0.5
                   for i, shape in enumerate(uniShape[1:])]


class NormalizedXavierWBInitializer(WBInitializer):
    def __init__(self, he: tp.Union["int", "float"] = 6):
        super(NormalizedXavierWBInitializer, self).__init__()
        self.he = he

    def _initialize(self, shape: "Shape") -> tp.Tuple["np.ndarray", "np.ndarray"]:
        biases = [self.sn((shape[i], 1), dtype=np.float32) * (self.he / (shape[i - 1] + shape[i])) ** 0.5
                  for i in range(1, shape.LAYERS)]
        weights = [self.sn((shape[i], shape[i - 1]), dtype=np.float32) * (self.he / (shape[i - 1] + shape[i])) ** 0.5
                   for i in range(1, shape.LAYERS)]

        return np.NONE + biases, np.NONE + weights
