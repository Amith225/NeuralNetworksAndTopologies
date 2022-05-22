from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..Topologies import *

import numpy as np

from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape, Network
from ..tools import Collections


class ConvShape(BaseShape):
    """

    """

    def __init__(self, inputShape: Union[tuple[int, int], tuple[int, int, int]],
                 *shapes: Union[tuple[int, int], tuple[int, int, int]]):
        super(ConvShape, self).__init__(inputShape, *shapes)

    @staticmethod
    def _formatShapes(shapes) -> tuple:
        formattedShape = []
        ellipsis_ = False
        for s in shapes:
            if s == Ellipsis: ellipsis_ = True; continue
            lenS = None
            assert isinstance(s, tuple) and 2 <= (lenS := len(s)) <= 3, \
                "all args of *shapes must be tuple of length 2 or 3"
            if lenS == 2: s = *s, s[-1]
            if ellipsis_: formattedShape[-1] += (s,); ellipsis_ = False; continue
            formattedShape.append((s,))
        return tuple(formattedShape)


def formatStride(stride) -> tuple:
    if isinstance(stride, int):
        assert stride > 0, "integer args of *strides must be > 0"
        return stride, stride
    assert isinstance(stride, tuple) and len(stride) == 2 and all(isinstance(s, int) for s in stride), \
        "non integer args of *strides must be integer tuple of length == 2"
    return stride


class Pooling(Collections):
    def __init__(self, *pooling: "Pooling.Base"):
        super(Pooling, self).__init__(*pooling)

    class Base:
        def __init__(self, stride: Union[int, tuple[int, int]]):
            self.stride = formatStride(stride)

    class MAX(Base): pass

    class MEAN(Base): pass


class Correlation(Collections):
    def __init__(self, *correlation: "Correlation.Base"):
        super(Correlation, self).__init__(*correlation)

    class Base:
        def __init__(self, stride: Union[int, tuple[int, int]]):
            self.stride = formatStride(stride)

    class VALID(Base): pass

    class FULL(Base): pass

    class SAME(Base): pass


class ConvLayer(BaseLayer):
    """

    """

    def __repr__(self):
        return super(ConvLayer, self).__repr__()

    def _initializeDepOptimizer(self):
        self.kernelOptimizer = self.optimizer.__new_copy__()
        self.biasesOptimizer = self.optimizer.__new_copy__()

    def _defineDeps(self, correlation: "Correlation.Base" = None, pooling: "Pooling.Base" = None) -> list['str']:
        if correlation is None: Correlation.VALID(1)
        if pooling is None: pooling = Pooling.MAX(1)
        self.pooling = pooling
        self.correlation = correlation
        # todo: how will shape be?
        self.kernels = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *self.SHAPE.HIDDEN, self.SHAPE.OUTPUT))
        self.biases = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *self.SHAPE.OUTPUT, self.SHAPE.OUTPUT))
        return ["kernels", "biases"]

    def _fire(self) -> "np.ndarray":
        pass

    def _wire(self) -> "np.ndarray":
        pass


class ConvPlot(BasePlot):
    """

    """


class ConvNN:
    pass
