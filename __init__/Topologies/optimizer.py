from abc import ABCMeta, abstractmethod

import numpy as np


class BaseOptimizer(metaclass=ABCMeta):
    __args, __kwargs = (), {}

    def __new__(cls, *args, **kwargs):
        cls.__args, cls.__kwargs = args, kwargs
        obj = super(BaseOptimizer, cls).__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    @classmethod
    def __new_copy__(cls):
        return cls.__new__(cls, *cls.__args, *cls.__kwargs)

    def __init__(self, learningRate=.01, alpha=.9, beta=.09, epsilon=1e-8):
        self.LEARNING_RATE = np.float32(learningRate)
        self.EPSILON = np.float32(epsilon)
        self.ALPHA = np.float32(alpha)
        self.ALPHA_BAR = 1 - self.ALPHA
        self.BETA = np.float32(beta)
        self.BETA_BAR = 1 - self.BETA

    def __call__(self, delta):
        return self._optimize(delta)

    @abstractmethod
    def _optimize(self, delta) -> "np.ndarray":
        pass


class GradientDecent(BaseOptimizer):
    def _optimize(self, delta) -> "np.ndarray":
        return delta * self.LEARNING_RATE


class Decay:
    pass


class Momentum:
    pass


class NesterovMomentum:
    pass


class AdaGrad:
    pass


class RmpProp:
    pass


class AdaDelta:
    pass
