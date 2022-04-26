class Base:
    from .base import BaseShape as Shape, BaseLayer as Layer, BasePlot as PLot, BaseNN as NN, UniversalShape


class Dense:
    from .dense import DenseShape as Shape, DenseLayer as Layer, DensePlot as Plot, DenseNN as NN


__all__ = [
    "Base", "Dense"
]
