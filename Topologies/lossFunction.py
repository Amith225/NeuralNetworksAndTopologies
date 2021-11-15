import typing as _tp

import numpy as _np

if _tp.TYPE_CHECKING:
    from . import *
    from NeuralNetworks import *


class LossFunction:
    def __init__(self, lossFunction):
        self.__eval = lossFunction

    def eval(self, output, target):
        return self.__eval(output, target)

    @staticmethod
    def meanSquare():
        def lossFunction(output, target):
            loss = output - target

            return _np.einsum('lij,lij->', loss, loss), loss

        return LossFunction(lossFunction)
