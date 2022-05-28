from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..NeuralNetworks import *


class BaseInitializer(metaclass=ABCMeta):
    rnd = np.random.default_rng()

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __new__(cls, *args, **kwargs):
        cls.RAW_ARGS = args
        cls.RAW_KWARGS = kwargs
        return super(BaseInitializer, cls).__new__(cls)

    def __save__(self) -> tuple["str", "tuple", "dict"]:
        return self.__class__.__name__, self.RAW_ARGS, self.RAW_KWARGS

    @staticmethod
    def __load__(name, raw_args, raw_kwargs) -> "BaseInitializer":
        return globals()[name](*raw_args, **raw_kwargs)

    def __call__(self, shape: "Base.Shape") -> "np.ndarray":
        """
        :param shape: a UniversalShape with UniversalShape(...).INPUT as the lower layer,
        UniversalShape(...).HIDDEN as the desired shape for initialization,
        UniversalShape(...).OUTPUT as the higher layer
        :return: 
        """
        return self._initialize(shape)

    @abstractmethod
    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        pass


class Uniform(BaseInitializer):
    def __repr__(self):
        start = self.start
        stop = self.stop
        return f"{super(Uniform, self).__repr__()[:-1]}: {start=}: {stop}>"

    def __init__(self, start: "float" = -1, stop: "float" = 1):
        self.start = start
        self.stop = stop

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.uniform(self.start, self.stop, shape.HIDDEN).astype(dtype=np.float32)


class Normal(BaseInitializer):
    def __repr__(self):
        scale = self.scale
        return f"{super(Normal, self).__repr__()[:-1]}: {scale=}>"

    def __init__(self, scale: "float" = 1):
        self.scale = scale

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * self.scale


class Xavier(BaseInitializer):
    def __repr__(self):
        he = self.he
        return f"{super(Xavier, self).__repr__()[:-1]}: {he=}>"

    def __init__(self, he: "float" = 1):
        self.he = he

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * (self.he / np.prod(shape.INPUT)) ** 0.5


class NormalizedXavier(BaseInitializer):
    def __repr__(self):
        he = self.he
        return f"{super(NormalizedXavier, self).__repr__()[:-1]}: {he=}>"

    def __init__(self, he: "float" = 6):
        self.he = he

    def _initialize(self, shape: "Base.Shape"):
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * (
                self.he / (np.prod(shape.INPUT) + np.prod(shape.OUTPUT)) ** (
                2 / (len(shape.INPUT) + len(shape.OUTPUT)))) ** 0.5
