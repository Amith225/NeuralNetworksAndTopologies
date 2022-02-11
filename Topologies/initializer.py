import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..NeuralNetworks import _
    from ..Utils import _
from abc import ABCMeta, abstractmethod

import numpy as np

# setup list or element numpy array of None
np.NONE = [np.array([None])]


class Initializer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.sn = np.random.default_rng().standard_normal

    def __call__(self, uniShape):
        return self._initialize(uniShape)

    @abstractmethod
    def _initialize(self, uniShape):
        pass


class Uniform(Initializer):
    def __init__(self, start: tp.Union["int", "float"] = -1, stop: tp.Union["int", "float"] = 1):
        super(Uniform, self).__init__()
        self.start = start
        self.stop = stop

    def _initialize(self, uniShape):
        return np.NONE + [np.random.uniform(self.start, self.stop, shape).astype(dtype=np.float32)
                          for shape in uniShape[1:]]


class Normal(Initializer):
    def __init__(self, scale: tp.Union["int", "float"] = 1):
        super(Normal, self).__init__()
        self.scale = scale

    def _initialize(self, uniShape):
        return np.NONE + [self.sn(shape, dtype=np.float32) * self.scale for shape in uniShape[1:]]


# fixme: needs improvement in np.prod
class Xavier(Initializer):
    def __init__(self, he: tp.Union["int", "float"] = 1):
        super(Xavier, self).__init__()
        self.he = he

    def _initialize(self, uniShape):
        return np.NONE + [self.sn(shape, dtype=np.float32) *
                          (self.he / np.prod(uniShape[i - 1])**(1/len(uniShape[i - 1]))) ** 0.5
                          for i, shape in enumerate(uniShape[1:])]


class NormalizedXavier(Initializer):
    def __init__(self, he: tp.Union["int", "float"] = 6):
        super(NormalizedXavier, self).__init__()
        self.he = he

    def _initialize(self, uniShape):
        return np.NONE + [self.sn(shape, dtype=np.float32) *
                          (self.he /
                           (np.prod(uniShape[i - 1]) + np.prod(shape))**(2/(len(uniShape[i - 1]) + len(shape)))) ** 0.5
                          for i, shape in enumerate(uniShape[1:])]
