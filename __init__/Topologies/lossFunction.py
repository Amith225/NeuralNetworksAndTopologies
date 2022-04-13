from abc import ABCMeta, abstractmethod


class AbstractLossFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, output, target):
        return self._eval(output, target)

    @abstractmethod
    def _eval(self, output, target):
        pass


class MeanSquareLossFunction(AbstractLossFunction):
    def __init__(self):
        super(MeanSquareLossFunction, self).__init__()

    def _eval(self, output, target):
        loss = output - target

        return (loss * loss).sum(), loss
