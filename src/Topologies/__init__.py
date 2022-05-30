from ..tools import Collections, DunderSaveLoad
from .dataBase import DataBase, PlotDataBase
from .activationFunction import BaseActivationFunction, \
        Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus
from .initializer import BaseInitializer, \
        Uniform, Normal, Xavier, NormalizedXavier
from .optimizer import BaseOptimizer, \
        GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam
from .lossFunction import BaseLossFunction, \
        MeanSquare, CrossEntropy


class Activators(Collections, DunderSaveLoad):
    Base, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus = \
        BaseActivationFunction, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus

    def __init__(self, *activationFunctions: "Activators.Base"):
        super(Activators, self).__init__(*activationFunctions)


class Initializers(Collections, DunderSaveLoad):
    Base, Uniform, Normal, Xavier, NormalizedXavier = \
        BaseInitializer, Uniform, Normal, Xavier, NormalizedXavier

    def __init__(self, *initializer: "Initializers.Base"):  # noqa
        super(Initializers, self).__init__(*initializer)


class Optimizers(Collections, DunderSaveLoad):
    Base, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmpProp, AdaDelta, Adam = \
        BaseOptimizer, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam

    def __init__(self, *optimizers: "Optimizers.Base"):
        super(Optimizers, self).__init__(*optimizers)


class LossFunction:
    Base, MeanSquare, CrossEntropy = \
        BaseLossFunction, MeanSquare, CrossEntropy


__all__ = [
    "Activators", "Initializers", "LossFunction", "Optimizers",
    "DataBase", "PlotDataBase"
]
