from typing import Type

from .base import BaseShape, BaseLayer, BasePlot, BaseNN, \
    UniversalShape, Network
from .dense import DenseShape, DenseLayer, DensePlot, DenseNN
from .conv import ConvShape, ConvLayer, ConvPlot, ConvNN, Correlation, Pooling


class Base:
    Shape: Type["BaseShape"] = BaseShape
    Layer: Type["BaseLayer"] = BaseLayer
    Plot: Type["BasePlot"] = BasePlot
    NN: Type["BaseNN"] = BaseNN


class Dense:
    Shape: Type["DenseShape"] = DenseShape
    Layer: Type["DenseLayer"] = DenseLayer
    Plot: Type["DensePlot"] = DensePlot
    NN: Type["DenseNN"] = DenseNN


class Conv:
    Shape: Type["ConvShape"] = ConvShape
    Layer: Type["ConvLayer"] = ConvLayer
    Plot: Type["ConvPlot"] = ConvPlot
    NN: Type["ConvNN"] = ConvNN


__all__ = [
    "Base", "Dense", "Conv",
    "UniversalShape", "Network",
    "Pooling", "Correlation"
]
