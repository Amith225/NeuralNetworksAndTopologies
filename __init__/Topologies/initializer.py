from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..NeuralNetworks import *

import numpy as np


class BaseInitializer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.rnd = np.random.default_rng()

    def __call__(self, shape: "Base.Shape") -> "np.ndarray":
        return self._initialize(shape)

    @abstractmethod
    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        pass


class Uniform(BaseInitializer):
    def __init__(self, start: "float" = -1, stop: "float" = 1):
        super(Uniform, self).__init__()
        self.start = start
        self.stop = stop

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.uniform(self.start, self.stop, shape.HIDDEN).astype(dtype=np.float32)


class Normal(BaseInitializer):
    def __init__(self, scale: "float" = 1):
        super(Normal, self).__init__()
        self.scale = scale

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * self.scale


class Xavier(BaseInitializer):
    def __init__(self, he: "float" = 1):
        super(Xavier, self).__init__()
        self.he = he

    # fixme: needs improvement in np.prod
    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * (
                self.he / np.prod(shape.INPUT) ** (1 / len(shape.INPUT))) ** 0.5


class NormalizedXavier(BaseInitializer):
    def __init__(self, he: "float" = 6):
        super(NormalizedXavier, self).__init__()
        self.he = he

    def _initialize(self, shape: "Base.Shape"):
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * (
                    self.he / (np.prod(shape.INPUT) + np.prod(shape.OUTPUT)) ** (
                        2 / (len(shape.INPUT) + len(shape.OUTPUT)))) ** 0.5
