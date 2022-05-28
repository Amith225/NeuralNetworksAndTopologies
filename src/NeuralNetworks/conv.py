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
        def __repr__(self):
            return f"<{self.__class__.__name__}:{self.stride}:{self.shape}>"

        def __init__(self, shape: Union[int, tuple[int, int]] = None, stride: Union[int, tuple[int, int]] = None):
            if shape is None: shape = 2
            self.shape = formatStride(shape)
            if stride is None: stride = self.shape
            self.stride = formatStride(stride)

    class MAX(Base): pass

    class MEAN(Base): pass


class Correlation(Collections):
    def __init__(self, *correlation: "Correlation.Base"):
        super(Correlation, self).__init__(*correlation)

    class Base:
        def __repr__(self):
            return f"<{self.__class__.__name__}:{self.stride}>"

        def __init__(self, stride: Union[int, tuple[int, int]] = None):
            if stride is None: stride = 1
            self.stride = formatStride(stride)

    class VALID(Base): pass

    class FULL(Base): pass

    class SAME(Base): pass


class ConvLayer(BaseLayer):
    """

    """

    def _initializeDepOptimizer(self):
        self.kernelOptimizer = self.optimizer.__new_copy__()
        self.biasesOptimizer = self.optimizer.__new_copy__()

    def _defineDeps(self, correlation: "Correlation.Base" = None, pooling: "Pooling.Base" = None) -> list['str']:
        if correlation is None: correlation = Correlation.VALID()
        if pooling is None: pooling = Pooling.MAX()
        self.pooling = pooling
        self.correlation = correlation
        self.kernelShape, self.kernelPad, self.kernelOut = \
            self.__findKernelPadOut(self.SHAPE.INPUT, self.SHAPE.OUTPUT, self.correlation.__class__,
                                    self.correlation.stride)
        _, self.poolPad, self.poolOut = \
            self.__findKernelPadOut(self.kernelOut, (self.kernelShape[0], *self.pooling.shape),
                                    self.correlation.__class__, self.pooling.stride)
        self.kernel = self.INITIALIZER(UniversalShape(self.SHAPE.HIDDEN, *self.kernelShape, self.SHAPE.OUTPUT))
        self.biases = self.INITIALIZER(UniversalShape(self.SHAPE.HIDDEN, *self.poolOut, self.SHAPE.OUTPUT))
        self.delta = None
        self.activeDerivedDelta = None
        return ["kernel", "biases"]

    def _fire(self) -> "np.ndarray":
        pass

    def _wire(self) -> "np.ndarray":
        pass

    @staticmethod
    def __findKernelPadOut(kernelInput, kernelBaseShape, correlation, stride):
        kernel = kernelBaseShape[0], kernelInput[0], *kernelBaseShape[1:]
        if correlation is Correlation.VALID:
            out = np.ceil((np.array(kernelInput[-2:]) - kernel[-2:]) / stride) + 1
        elif correlation is Correlation.FULL:
            out = np.ceil((np.array(kernelInput[-2:]) - kernel[-2:] +
                           2 * (kernel[-2:] - np.int16(1))) / stride) + 1  # noqa
        elif correlation is Correlation.SAME:
            out = np.array(kernelInput[-2:])
        else:
            raise ValueError("Invalid correlation type")
        pad = ConvLayer.__findPadFromOut(kernelInput, out, stride, kernel)
        out = kernel[0], *out.astype(np.int16)
        return kernel, tuple(pad.tolist()), out

    @staticmethod
    def __findPadFromOut(inp, out, stride, kernel):
        return ((out - 1) * stride + kernel[-2:] - inp[-2:]).astype(np.int16)


class ConvPlot(BasePlot):
    """

    """


class ConvNN:
    pass
