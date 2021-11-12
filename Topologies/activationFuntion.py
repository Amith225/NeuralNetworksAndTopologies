# library direct imports
from typing import *

# library imports
import numpy as np


class WBActivationFunction:  # main class
    # WBActivationFunction class and methods / params for custom activation function
    def __init__(self, activation: "Callable", activatedDerivative: "Callable"):
        self.__activation = activation
        self.__activatedDerivative = activatedDerivative

    def activation(self, x: np.ndarray) -> np.ndarray:
        return self.__activation(x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.__activatedDerivative(activatedX)

    # Pre-Defined activation functions
    @staticmethod
    def sigmoid(smooth: Union[int, float] = 1, offset: Union[int, float] = 0) -> "WBActivationFunction":
        # constants
        ONE = np.float32(1)
        E = np.float32(np.e)
        SMOOTH = np.float32(smooth)
        OFFSET = np.float32(offset)

        def activation(x: np.ndarray) -> np.ndarray:
            return ONE / (ONE + E ** (-SMOOTH * (x - OFFSET)))

        def activatedDerivative(activatedX: np.ndarray) -> np.ndarray:
            return SMOOTH * (activatedX * (ONE - activatedX))

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def tanh(alpha: Union[int, float] = 1) -> "WBActivationFunction":
        # constants
        ALPHA = np.float32(alpha)

        def activation(x: np.ndarray) -> np.ndarray: return np.arctan(ALPHA * x)

        def activatedDerivative(activatedX: np.ndarray) -> np.ndarray: return ALPHA * np.square(np.cos(activatedX))

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def relu() -> "WBActivationFunction":
        # constants
        ONE = np.float32(1)

        def activation(x: np.ndarray) -> np.ndarray: return x * (x > 0)

        def activatedDerivative(activatedX: np.ndarray) -> np.ndarray: return ONE * (activatedX != 0)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def prelu(leak: Union[int, float] = 0.01) -> "WBActivationFunction":
        # constants
        if leak < 0:
            raise ValueError("parameter 'leak' cannot be less than zero")
        ONE = np.float32(1)
        LEAK = np.float32(leak)

        def activation(x: np.ndarray) -> np.ndarray: return np.where(x > 0, x, LEAK * x)

        def activatedDerivative(activatedX: np.ndarray) -> np.ndarray: return np.where(activatedX <= 0, LEAK, ONE)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def elu(alpha: Union[int, float] = 1) -> "WBActivationFunction":
        # constants
        if alpha < 0:
            raise ValueError("parameter 'alpha' cannot be less than zero")
        ONE = np.float32(1)
        E = np.e
        ALPHA = np.float32(alpha)

        def activation(x: np.ndarray) -> np.ndarray: return np.where(x > 0, x, ALPHA * (E ** x - 1))

        def activatedDerivative(activatedX: np.ndarray) -> np.ndarray:
            return np.where(activatedX <= 0, activatedX + ALPHA, ONE)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def softmax() -> "WBActivationFunction":
        # constants
        E = np.float32(np.e)

        def activation(x: np.ndarray) -> np.ndarray:
            numerator = E ** (x - x.max(axis=1, keepdims=1))

            return numerator / numerator.sum(axis=1, keepdims=1)

        def activatedDerivative(activatedX: np.ndarray):
            jacobian = activatedX @ activatedX.transpose([0, 2, 1])
            diagIndexes = np.diag_indices(jacobian.shape[1])
            jacobian[:, [diagIndexes[1]], [diagIndexes[0]]] = (activatedX * (1 - activatedX)).transpose(0, 2, 1)

            return jacobian.sum(axis=2, keepdims=1)

        return WBActivationFunction(activation, activatedDerivative)

    @staticmethod
    def softplus() -> "WBActivationFunction":
        # constants
        E = np.float32(np.e)
        ONE = np.float32(1)

        def activation(x: np.ndarray) -> np.ndarray: return np.log(ONE + E ** x)

        def activatedDerivative(activatedX: np.ndarray) -> np.ndarray: return ONE - E ** -activatedX

        return WBActivationFunction(activation, activatedDerivative)
