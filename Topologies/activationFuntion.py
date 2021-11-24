import typing as _tp
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np

if _tp.TYPE_CHECKING:
    pass


class AbstractActivationFunction(metaclass=_ABCMeta):  # main class
    # AbstractActivationFunction class and methods
    @_abstractmethod
    def __init__(self, *args, **kwargs):
        # constants
        self.ONE = _np.float32(1)
        self.E = _np.float32(_np.e)

    @_abstractmethod
    def activation(self, x: _np.ndarray) -> "_np.ndarray":
        pass

    @_abstractmethod
    def activatedDerivative(self, activatedX: _np.ndarray) -> "_np.ndarray":
        pass


class Sigmoid(AbstractActivationFunction):
    def __init__(self, smooth: _tp.Union[int, float] = 1, offset: _tp.Union[int, float] = 0):
        super(Sigmoid, self).__init__()
        # constants
        self.SMOOTH = _np.float32(smooth)
        self.OFFSET = _np.float32(offset)

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return self.ONE / (self.ONE + self.E ** (-self.SMOOTH * (x - self.OFFSET)))

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return self.SMOOTH * (activatedX * (self.ONE - activatedX))


class Tanh(AbstractActivationFunction):
    def __init__(self, alpha: _tp.Union[int, float] = 1):
        super(Tanh, self).__init__()
        # constants
        self.ALPHA = _np.float32(alpha)

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return _np.arctan(self.ALPHA * x)

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return self.ALPHA * _np.square(_np.cos(activatedX))


class Relu(AbstractActivationFunction):
    def __init__(self):
        super(Relu, self).__init__()

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return x * (x > 0)

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return self.ONE * (activatedX != 0)


class Prelu(AbstractActivationFunction):
    def __init__(self, leak: _tp.Union[int, float] = 0.01):
        super(Prelu, self).__init__()
        if leak < 0:
            raise ValueError("parameter 'leak' cannot be less than zero")
        # constants
        self.LEAK = _np.float32(leak)

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return _np.where(x > 0, x, self.LEAK * x)

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return _np.where(activatedX <= 0, self.LEAK, self.ONE)


class Elu(AbstractActivationFunction):
    def __init__(self, alpha: _tp.Union[int, float] = 1):
        super(Elu, self).__init__()
        if alpha < 0:
            raise ValueError("parameter 'alpha' cannot be less than zero")
        # constants
        self.ALPHA = _np.float32(alpha)

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return _np.where(x > 0, x, self.ALPHA * (self.E ** x - 1))

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return _np.where(activatedX <= 0, activatedX + self.ALPHA, self.ONE)


class Softmax(AbstractActivationFunction):
    def __init__(self):
        super(Softmax, self).__init__()
        self.__jacobian = None

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        numerator = self.E ** (x - x.max(axis=1, keepdims=1))

        return numerator / numerator.sum(axis=1, keepdims=1)

    def activatedDerivative(self, activatedX: _np.ndarray):
        self.__jacobian = (ax := activatedX.transpose([0, 2, 1])[:, :, :, None]) @ ax.transpose([0, 1, 3, 2])
        diagIndexes = _np.diag_indices(self.__jacobian.shape[2])
        self.__jacobian[:, :, [diagIndexes[1]], [diagIndexes[0]]] = \
            (activatedX * (1 - activatedX)).transpose(0, 2, 1)[:, :, None]

        return self.__jacobian.sum(axis=3).transpose(0, 2, 1)


class Softplus(AbstractActivationFunction):
    def __init__(self):
        super(Softplus, self).__init__()

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return _np.log(self.ONE + self.E ** x)

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return self.ONE - self.E ** -activatedX
