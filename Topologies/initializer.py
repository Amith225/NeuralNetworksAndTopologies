# library direct imports
import typing as _tp

# library imports
import numpy as _np

# library imports for type checking
if _tp.TYPE_CHECKING:
    from . import *
    from NeuralNetworks import *

# setup list or element numpy array of None
_np.NONE = [_np.array([None])]


class WBInitializer:  # main class
    def __init__(self, initializer, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.__initializer = initializer

    def initialize(self, shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
        return self.__initializer(shape)

    @staticmethod
    def uniform(start: _tp.Union["int", "float"] = -1, stop: _tp.Union["int", "float"] = 1) -> "WBInitializer":
        def initialize(shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
            biases = [_np.random.uniform(start, stop, (shape[i], 1)).astype(dtype=_np.float32)
                      for i in range(1, shape.LAYERS)]
            weights = [_np.random.uniform(start, stop, (shape[i], shape[i - 1])).astype(dtype=_np.float32)
                       for i in range(1, shape.LAYERS)]

            return _np.NONE + biases, _np.NONE + weights

        return WBInitializer(initialize)

    @staticmethod
    def normal(scale: _tp.Union["int", "float"] = 1):
        def initialize(shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
            sn = _np.random.default_rng().standard_normal
            biases = [(sn((shape[i], 1), dtype=_np.float32)) * scale for i in range(1, shape.LAYERS)]
            weights = [(sn((shape[i], shape[i - 1]), dtype=_np.float32)) * scale for i in range(1, shape.LAYERS)]

            return _np.NONE + biases, _np.NONE + weights

        return WBInitializer(initialize)

    @staticmethod
    def xavier(he: _tp.Union["int", "float"] = 1):
        def initialize(shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
            sn = _np.random.default_rng().standard_normal
            biases = [sn((shape[i], 1), dtype=_np.float32) * (he / shape[i - 1]) ** 0.5 for i in range(1, shape.LAYERS)]
            weights = [sn((shape[i], shape[i - 1]), dtype=_np.float32) * (he / shape[i - 1]) ** 0.5
                       for i in range(1, shape.LAYERS)]

            return _np.NONE + biases, _np.NONE + weights

        return WBInitializer(initialize)

    @staticmethod
    def normalizedXavier(he: _tp.Union["int", "float"] = 6):
        def initialize(shape: "WBShape") -> _tp.Tuple["_np.ndarray", "_np.ndarray"]:
            sn = _np.random.default_rng().standard_normal
            biases = [sn((shape[i], 1), dtype=_np.float32) * (he / (shape[i - 1] + shape[i])) ** 0.5
                      for i in range(1, shape.LAYERS)]
            weights = [sn((shape[i], shape[i - 1]), dtype=_np.float32) * (he / (shape[i - 1] + shape[i])) ** 0.5
                       for i in range(1, shape.LAYERS)]

            return _np.NONE + biases, _np.NONE + weights

        return WBInitializer(initialize)
