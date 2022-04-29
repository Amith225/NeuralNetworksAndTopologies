from ..tools import Collections
from .dataBase import DataBase, PlotDataBase


class Activators(Collections):
    from .activationFunction import BaseActivationFunction as Base, \
        Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus
    Base, Sigmoid, Tanh, Relu, Prelu, Elu, Softmax, Softplus = \
        Base, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus

    def __init__(self, *activationFunctions: "Activators.Base"):
        super(Activators, self).__init__(*activationFunctions)


class Initializers(Collections):
    from .initializer import BaseInitializer as Base, \
        Uniform, Normal, Xavier, NormalizedXavier
    Base, Uniform, Normal, Xavier, NormalizedXavier = \
        Base, Uniform, Normal, Xavier, NormalizedXavier

    def __init__(self, *initializer: "Initializers.Base"):
        super(Initializers, self).__init__(*initializer)


class Optimizers(Collections):
    from .optimizer import BaseOptimizer as Base, \
        GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam
    Base, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmpProp, AdaDelta, Adam = \
        Base, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam

    def __init__(self, *optimizers: "Optimizers.Base"):
        super(Optimizers, self).__init__(*optimizers)


class LossFunction:
    from .lossFunction import BaseLossFunction as Base, \
        MeanSquare
    Base, MeanSquare = Base, MeanSquare


__all__ = [
    "Activators", "Initializers", "LossFunction", "Optimizers",
    "DataBase", "PlotDataBase"
]
