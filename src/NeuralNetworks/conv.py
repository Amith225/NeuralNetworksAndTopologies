from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..Topologies import *

import numpy as np

from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape, Network


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


class ConvLayer(BaseLayer):
    """

    """

    def __repr__(self):
        return super(ConvLayer, self).__repr__()

    def _initializeDepOptimizer(self):
        pass

    def _defineDeps(self, *depArgs, **depKwargs) -> list['str']:
        pass

    def _fire(self) -> "np.ndarray":
        pass

    def _wire(self) -> "np.ndarray":
        pass


class ConvPlot(BasePlot):
    """

    """


class ConvNN:
    pass
