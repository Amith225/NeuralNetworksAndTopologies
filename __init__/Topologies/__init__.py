from .dataBase import DataBase, PlotDataBase


class ActivationFunction:
    from .activationFunction import AbstractActivationFunction as Abstract, \
        Sigmoid, Tanh, Relu, Prelu, Elu, Softmax, Softplus


class Initializer:
    from .initializer import BaseInitializer as Base, \
        Uniform, Normal, Xavier, NormalizedXavier


class LossFunction:
    from .lossFunction import AbstractLossFunction as Abstract, \
        MeanSquare


class Optimizer:
    from .optimizer import BaseOptimizer as Base, \
        GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmpProp, AdaDelta


__all__ = [
    "ActivationFunction", "Initializer", "LossFunction", "Optimizer",
    "DataBase", "PlotDataBase"
]
