from typing import *
from abc import ABCMeta, abstractmethod

import numpy as np


class BaseActivationFunction(metaclass=ABCMeta):
    ONE = np.float32(1)
    E = np.float32(np.e)

    @abstractmethod
    def activation(self, x: np.ndarray) -> "np.ndarray":
        pass

    @abstractmethod
    def activatedDerivative(self, activatedX: np.ndarray) -> "np.ndarray":
        pass


class Sigmoid(BaseActivationFunction):
    def __init__(self, smooth: Union[int, float] = 1, offset: Union[int, float] = 0):
        self.SMOOTH = np.float32(smooth)
        self.OFFSET = np.float32(offset)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return self.ONE / (self.ONE + self.E ** (-self.SMOOTH * (x - self.OFFSET)))

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.SMOOTH * (activatedX * (self.ONE - activatedX))


class Tanh(BaseActivationFunction):
    def __init__(self, alpha: Union[int, float] = 1):
        self.ALPHA = np.float32(alpha)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.arctan(self.ALPHA * x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ALPHA * np.square(np.cos(activatedX))


class Relu(BaseActivationFunction):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ONE * (activatedX != 0)


class Prelu(BaseActivationFunction):
    def __init__(self, leak: Union[int, float] = 0.01):
        if leak < 0: raise ValueError("parameter 'leak' cannot be less than zero")
        self.LEAK = np.float32(leak)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.LEAK * x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return np.where(activatedX <= 0, self.LEAK, self.ONE)


class Elu(BaseActivationFunction):
    def __init__(self, alpha: Union[int, float] = 1):
        if alpha < 0: raise ValueError("parameter 'alpha' cannot be less than zero")
        self.ALPHA = np.float32(alpha)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.ALPHA * (self.E ** x - 1))

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return np.where(activatedX <= 0, activatedX + self.ALPHA, self.ONE)


class Softmax(BaseActivationFunction):
    def __init__(self):
        self.__jacobian = None

    def activation(self, x: np.ndarray) -> np.ndarray:
        numerator = self.E ** (x - x.max(axis=-2, keepdims=True))
        return numerator / numerator.sum(axis=-2, keepdims=1)

    def activatedDerivative(self, activatedX: np.ndarray):
        self.__jacobian = np.einsum('...ji, ...ki -> ...ijk', activatedX, activatedX, optimize='greedy')
        diagIndexes = np.diag_indices(self.__jacobian.shape[-1])
        self.__jacobian[..., diagIndexes[0], diagIndexes[1]] = \
            (activatedX * (1 - activatedX)).transpose().reshape(self.__jacobian.shape[:-1])
        return self.__jacobian.sum(axis=-1).transpose().reshape(activatedX.shape)


class Softplus(BaseActivationFunction):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.ONE + self.E ** x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ONE - self.E ** -activatedX
