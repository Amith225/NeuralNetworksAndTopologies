import typing as tp
if tp.TYPE_CHECKING:
    from utils import *
    from . import *
    from ..NeuralNetworks import *

from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod


class LossFunction(metaclass=_ABCMeta):
    @_abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, output, target):
        return self._eval(output, target)

    @_abstractmethod
    def _eval(self, output, target):
        pass


class MeanSquareLossFunction(LossFunction):
    def __init__(self):
        super(MeanSquareLossFunction, self).__init__()

    def _eval(self, output, target):
        loss = output - target

        return (loss * loss).sum(), loss
