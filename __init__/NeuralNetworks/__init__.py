class Base:
    from .base import BaseShape as Shape, BaseLayer as Layer, BasePlot as Plot, BaseNN as NN, \
        UniversalShape, Network
    Shape, Layer, Plot, NN = Shape, Layer, Plot, NN
    UniversalShape, Network = UniversalShape, Network


class Dense:
    from .dense import DenseShape as Shape, DenseLayer as Layer, DensePlot as Plot, DenseNN as NN
    Shape, Layer, Plot, NN = Shape, Layer, Plot, NN


class Conv:
    from .conv import ConvShape as Shape, ConvLayer as Layer, ConvPlot as Plot, ConvNN as NN
    Shape, Layer, Plot, NN = Shape, Layer, Plot, NN


__all__ = [
    "Base", "Dense", "Conv"
]
