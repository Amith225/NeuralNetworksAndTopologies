class Base:
    from .base import BaseShape, BaseShape, BasePlot, BaseNN, \
        UniversalShape, Network
    Shape, Layer, Plot, NN = BaseShape, BaseShape, BasePlot, BaseNN
    UniversalShape, Network = UniversalShape, Network


class Dense:
    from .dense import DenseShape, DenseLayer, DensePlot, DenseNN
    Shape, Layer, Plot, NN = DenseShape, DenseLayer, DensePlot, DenseNN


class Conv:
    from .conv import ConvShape, ConvLayer, ConvPlot, ConvNN
    Shape, Layer, Plot, NN = ConvShape, ConvLayer, ConvPlot, ConvNN


__all__ = [
    "Base", "Dense", "Conv"
]
