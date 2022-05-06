from ..tools import Collections
from .dataBase import DataBase, PlotDataBase


class Activators(Collections):
    from .activationFunction import BaseActivationFunction, \
        Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus
    Base, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus = \
        BaseActivationFunction, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus

    def __init__(self, *activationFunctions: "Activators.Base"):
        super(Activators, self).__init__(*activationFunctions)


class Initializers(Collections):
    from .initializer import BaseInitializer, \
        Uniform, Normal, Xavier, NormalizedXavier
    Base, Uniform, Normal, Xavier, NormalizedXavier = \
        BaseInitializer, Uniform, Normal, Xavier, NormalizedXavier

    def __init__(self, *initializer: "Initializers.Base"):
        super(Initializers, self).__init__(*initializer)


class Optimizers(Collections):
    from .optimizer import BaseOptimizer, \
        GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam
    Base, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmpProp, AdaDelta, Adam = \
        BaseOptimizer, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam

    def __init__(self, *optimizers: "Optimizers.Base"):
        super(Optimizers, self).__init__(*optimizers)


class LossFunction:
    from .lossFunction import BaseLossFunction, \
        MeanSquare
    Base, MeanSquare = BaseLossFunction, MeanSquare


__all__ = [
    "Activators", "Initializers", "LossFunction", "Optimizers",
    "DataBase", "PlotDataBase"
]
