import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..NeuralNetworks import _
    from ..Utils import _
from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractActivationFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.ONE = np.float32(1)
        self.E = np.float32(np.e)

    @abstractmethod
    def activation(self, x: np.ndarray) -> "np.ndarray":
        pass

    @abstractmethod
    def activatedDerivative(self, activatedX: np.ndarray) -> "np.ndarray":
        pass


class Sigmoid(AbstractActivationFunction):
    def __init__(self, smooth: tp.Union[int, float] = 1, offset: tp.Union[int, float] = 0):
        super(Sigmoid, self).__init__()
        self.SMOOTH = np.float32(smooth)
        self.OFFSET = np.float32(offset)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return self.ONE / (self.ONE + self.E ** (-self.SMOOTH * (x - self.OFFSET)))

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.SMOOTH * (activatedX * (self.ONE - activatedX))


class Tanh(AbstractActivationFunction):
    def __init__(self, alpha: tp.Union[int, float] = 1):
        super(Tanh, self).__init__()
        self.ALPHA = np.float32(alpha)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.arctan(self.ALPHA * x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ALPHA * np.square(np.cos(activatedX))


class Relu(AbstractActivationFunction):
    def __init__(self):
        super(Relu, self).__init__()

    def activation(self, x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ONE * (activatedX != 0)


class Prelu(AbstractActivationFunction):
    def __init__(self, leak: tp.Union[int, float] = 0.01):
        super(Prelu, self).__init__()
        if leak < 0:
            raise ValueError("parameter 'leak' cannot be less than zero")
        self.LEAK = np.float32(leak)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.LEAK * x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return np.where(activatedX <= 0, self.LEAK, self.ONE)


class Elu(AbstractActivationFunction):
    def __init__(self, alpha: tp.Union[int, float] = 1):
        super(Elu, self).__init__()
        if alpha < 0:
            raise ValueError("parameter 'alpha' cannot be less than zero")
        self.ALPHA = np.float32(alpha)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.ALPHA * (self.E ** x - 1))

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return np.where(activatedX <= 0, activatedX + self.ALPHA, self.ONE)


class Softmax(AbstractActivationFunction):
    def __init__(self):
        super(Softmax, self).__init__()
        self.__jacobian = None

    def activation(self, x: np.ndarray) -> np.ndarray:
        numerator = self.E ** (x - x.max(axis=1, keepdims=1))

        return numerator / numerator.sum(axis=1, keepdims=1)

    def activatedDerivative(self, activatedX: np.ndarray):
        self.__jacobian = (ax := activatedX.transpose([0, 2, 1])[:, :, :, None]) @ ax.transpose([0, 1, 3, 2])
        diagIndexes = np.diag_indices(self.__jacobian.shape[2])
        self.__jacobian[:, :, [diagIndexes[1]], [diagIndexes[0]]] = \
            (activatedX * (1 - activatedX)).transpose(0, 2, 1)[:, :, None]

        return self.__jacobian.sum(axis=3).transpose(0, 2, 1)


class Softplus(AbstractActivationFunction):
    def __init__(self):
        super(Softplus, self).__init__()

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.ONE + self.E ** x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ONE - self.E ** -activatedX
