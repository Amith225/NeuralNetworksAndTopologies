# library direct imports
from typing import *

# library imports
import numpy as np

# library imports for type checking
if TYPE_CHECKING:
    from ..NeuralNetworks import *

# setup list or element numpy array of None
np.NONE = [np.array([None])]


class WBInitializer:  # main class
    def __init__(self, initializer, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.__initializer = initializer

    def initialize(self, shape: "WBShape") -> Tuple["np.ndarray", "np.ndarray"]:
        return self.__initializer(shape)

    @staticmethod
    def uniform(start: Union["int", "float"] = -1, stop: Union["int", "float"] = 1) -> "WBInitializer":
        def initialize(shape: "WBShape") -> Tuple["np.ndarray", "np.ndarray"]:
            biases = [np.random.uniform(start, stop, (shape[i], 1)).astype(dtype=np.float32)
                      for i in range(1, shape.LAYERS)]
            weights = [np.random.uniform(start, stop, (shape[i], shape[i - 1])).astype(dtype=np.float32)
                       for i in range(1, shape.LAYERS)]

            return np.NONE + biases, np.NONE + weights

        return WBInitializer(initialize)

    @staticmethod
    def normal(scale: Union["int", "float"] = 1):
        def initialize(shape: "WBShape") -> Tuple["np.ndarray", "np.ndarray"]:
            sn = np.random.default_rng().standard_normal
            biases = [(sn((shape[i], 1), dtype=np.float32)) * scale for i in range(1, shape.LAYERS)]
            weights = [(sn((shape[i], shape[i - 1]), dtype=np.float32)) * scale for i in range(1, shape.LAYERS)]

            return np.NONE + biases, np.NONE + weights

        return WBInitializer(initialize)

    @staticmethod
    def xavier(he: Union["int", "float"] = 1):
        def initialize(shape: "WBShape") -> Tuple["np.ndarray", "np.ndarray"]:
            sn = np.random.default_rng().standard_normal
            biases = [sn((shape[i], 1), dtype=np.float32) * (he / shape[i - 1]) ** 0.5 for i in range(1, shape.LAYERS)]
            weights = [sn((shape[i], shape[i - 1]), dtype=np.float32) * (he / shape[i - 1]) ** 0.5
                       for i in range(1, shape.LAYERS)]

            return np.NONE + biases, np.NONE + weights

        return WBInitializer(initialize)

    @staticmethod
    def normalizedXavier(he: Union["int", "float"] = 6):
        def initialize(shape: "WBShape") -> Tuple["np.ndarray", "np.ndarray"]:
            sn = np.random.default_rng().standard_normal
            biases = [sn((shape[i], 1), dtype=np.float32) * (he / (shape[i - 1] + shape[i])) ** 0.5
                      for i in range(1, shape.LAYERS)]
            weights = [sn((shape[i], shape[i - 1]), dtype=np.float32) * (he / (shape[i - 1] + shape[i])) ** 0.5
                       for i in range(1, shape.LAYERS)]

            return np.NONE + biases, np.NONE + weights

        return WBInitializer(initialize)
