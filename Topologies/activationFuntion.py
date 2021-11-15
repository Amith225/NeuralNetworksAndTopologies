# library direct imports
import typing as _tp

# library imports
import numpy as _np

if _tp.TYPE_CHECKING:
    from . import *
    from NeuralNetworks import *


class WBActivationFunction:  # main class
    # WBActivationFunction class and methods / params for custom activation function
    def __init__(self, activation: _tp.Callable, activatedDerivative: _tp.Callable):
        self.__activation = activation
        self.__activatedDerivative = activatedDerivative

    def activation(self, x: _np.ndarray) -> _np.ndarray:
        return self.__activation(x)

    def activatedDerivative(self, activatedX: _np.ndarray) -> _np.ndarray:
        return self.__activatedDerivative(activatedX)

    # Pre-Defined activation functions
    @staticmethod
    def sigmoid(smooth: _tp.Union[int, float] = 1, offset: _tp.Union[int, float] = 0) -> "WBActivationFunction":
        # constants
        ONE = _np.float32(1)
        E = _np.float32(_np.e)
        SMOOTH = _np.float32(smooth)
        OFFSET = _np.float32(offset)

        def activation(x: _np.ndarray) -> _np.ndarray:
            return ONE / (ONE + E ** (-SMOOTH * (x - OFFSET)))

        def activatedDerivative(activatedX: _np.ndarray) -> _np.ndarray:
            return SMOOTH * (activatedX * (ONE - activatedX))

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def tanh(alpha: _tp.Union[int, float] = 1) -> "WBActivationFunction":
        # constants
        ALPHA = _np.float32(alpha)

        def activation(x: _np.ndarray) -> _np.ndarray: return _np.arctan(ALPHA * x)

        def activatedDerivative(activatedX: _np.ndarray) -> _np.ndarray: return ALPHA * _np.square(_np.cos(activatedX))

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def relu() -> "WBActivationFunction":
        # constants
        ONE = _np.float32(1)

        def activation(x: _np.ndarray) -> _np.ndarray: return x * (x > 0)

        def activatedDerivative(activatedX: _np.ndarray) -> _np.ndarray: return ONE * (activatedX != 0)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def prelu(leak: _tp.Union[int, float] = 0.01) -> "WBActivationFunction":
        # constants
        if leak < 0:
            raise ValueError("parameter 'leak' cannot be less than zero")
        ONE = _np.float32(1)
        LEAK = _np.float32(leak)

        def activation(x: _np.ndarray) -> _np.ndarray: return _np.where(x > 0, x, LEAK * x)

        def activatedDerivative(activatedX: _np.ndarray) -> _np.ndarray: return _np.where(activatedX <= 0, LEAK, ONE)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def elu(alpha: _tp.Union[int, float] = 1) -> "WBActivationFunction":
        # constants
        if alpha < 0:
            raise ValueError("parameter 'alpha' cannot be less than zero")
        ONE = _np.float32(1)
        E = _np.e
        ALPHA = _np.float32(alpha)

        def activation(x: _np.ndarray) -> _np.ndarray: return _np.where(x > 0, x, ALPHA * (E ** x - 1))

        def activatedDerivative(activatedX: _np.ndarray) -> _np.ndarray:
            return _np.where(activatedX <= 0, activatedX + ALPHA, ONE)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def softmax() -> "WBActivationFunction":
        # constants
        E = _np.float32(_np.e)

        def activation(x: _np.ndarray) -> _np.ndarray:
            numerator = E ** (x - x.max(axis=1, keepdims=1))

            return numerator / numerator.sum(axis=1, keepdims=1)

        def activatedDerivative(activatedX: _np.ndarray):
            jacobian = activatedX @ activatedX.transpose([0, 2, 1])
            diagIndexes = _np.diag_indices(jacobian.shape[1])
            jacobian[:, [diagIndexes[1]], [diagIndexes[0]]] = (activatedX * (1 - activatedX)).transpose(0, 2, 1)

            return jacobian.sum(axis=2, keepdims=1)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def softplus() -> "WBActivationFunction":
        # constants
        E = _np.float32(_np.e)
        ONE = _np.float32(1)

        def activation(x: _np.ndarray) -> _np.ndarray: return _np.log(ONE + E ** x)

        def activatedDerivative(activatedX: _np.ndarray) -> _np.ndarray: return ONE - E ** -activatedX

        return WBActivationFunction(activation, activatedDerivative)
