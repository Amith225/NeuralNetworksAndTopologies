import typing as tp
if tp.TYPE_CHECKING:
    from . import _
    from ..NeuralNetworks import _
    from ..Utils import _
from abc import ABCMeta, abstractmethod


class LossFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, output, target):
        return self._eval(output, target)

    @abstractmethod
    def _eval(self, output, target):
        pass


class MeanSquareLossFunction(LossFunction):
    def __init__(self):
        super(MeanSquareLossFunction, self).__init__()

    def _eval(self, output, target):
        loss = output - target

        return (loss * loss).sum(), loss
