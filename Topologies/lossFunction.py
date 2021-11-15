import typing as _tp
from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

import numpy as _np

if _tp.TYPE_CHECKING:
    pass


class LossFunction(metaclass=_ABCMeta):
    @_abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @_abstractmethod
    def eval(self, output, target):
        pass


class MeanSquareLossFunction(LossFunction):
    def __init__(self):
        super(MeanSquareLossFunction, self).__init__()

    def eval(self, output, target):
        loss = output - target

        return _np.einsum('lij,lij->', loss, loss), loss
