from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractInitializer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.sn = np.random.default_rng().standard_normal

    def __call__(self, uniShape):
        return [np.NAN] + self._initialize(uniShape)

    @abstractmethod
    def _initialize(self, uniShape):
        pass


class Uniform(AbstractInitializer):
    def __init__(self, start: "float" = -1, stop: "float" = 1):
        super(Uniform, self).__init__()
        self.start = start
        self.stop = stop

    def _initialize(self, uniShape):
        return [np.random.uniform(self.start, self.stop, shape).astype(dtype=np.float32)
                for shape in uniShape[1:]]


class Normal(AbstractInitializer):
    def __init__(self, scale: "float" = 1):
        super(Normal, self).__init__()
        self.scale = scale

    def _initialize(self, uniShape):
        return [self.sn(shape, dtype=np.float32) * self.scale for shape in uniShape[1:]]


class Xavier(AbstractInitializer):
    def __init__(self, he: "float" = 1):
        super(Xavier, self).__init__()
        self.he = he

    # fixme: needs improvement in np.prod
    def _initialize(self, uniShape):
        return [self.sn(shape, dtype=np.float32) *
                (self.he / np.prod(uniShape[i - 1]) ** (1 / len(uniShape[i - 1]))) ** 0.5
                for i, shape in enumerate(uniShape[1:])]


class NormalizedXavier(AbstractInitializer):
    def __init__(self, he: "float" = 6):
        super(NormalizedXavier, self).__init__()
        self.he = he

    def _initialize(self, uniShape):
        return [self.sn(shape, dtype=np.float32) *
                (self.he /
                 (np.prod(uniShape[i - 1]) + np.prod(shape)) ** (2 / (len(uniShape[i - 1]) + len(shape)))) ** 0.5
                for i, shape in enumerate(uniShape[1:])]
