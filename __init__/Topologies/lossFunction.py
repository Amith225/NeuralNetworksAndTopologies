from abc import ABCMeta, abstractmethod


class BaseLossFunction(metaclass=ABCMeta):
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, output, target):
        return self._eval(output, target)

    @abstractmethod
    def _eval(self, output, target):
        pass


class MeanSquare(BaseLossFunction):
    def __init__(self):
        super(MeanSquare, self).__init__()

    def _eval(self, output, target):
        delta = output - target
        return (delta * delta).sum(axis=1).mean(), delta
