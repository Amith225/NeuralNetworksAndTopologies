from abc import ABCMeta, abstractmethod

import numpy as np


class BaseLossFunction(metaclass=ABCMeta):
    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __call__(self, output, target):
        return self._eval(output, target)

    @abstractmethod
    def _eval(self, output, target):
        pass


class MeanSquare(BaseLossFunction):
    def _eval(self, output, target):
        delta = output - target
        return (delta * delta).sum(axis=1).mean(), delta


class CrossEntropy(BaseLossFunction):
    def _eval(self, output, target):
        cross = -np.log(output) * target
        return (cross * cross).sum(axis=1).mean(), output - target
