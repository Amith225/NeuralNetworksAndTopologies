from utils import AbstractLoad, Plot
from NeuralNetworks.neuralNetwork import AbstractNeuralNetwork

import dill as dl
import numpy as np

import typing as tp

# library imports for type checking
if tp.TYPE_CHECKING:
    pass


class NeuralNetworkParserError(Exception):
    pass


class LoadNeuralNetwork(AbstractLoad):
    DEFAULT_DIR = AbstractNeuralNetwork.DEFAULT_DIR
    FILE_TYPE = AbstractNeuralNetwork.FILE_TYPE

    def __new__(cls, file, *args, **kwargs) -> "AbstractNeuralNetwork":
        return cls.load(file, *args, **kwargs)

    @classmethod
    def _read(cls, loadFile, *args, **kwargs) -> "AbstractNeuralNetwork":
        if AbstractNeuralNetwork not in (nn := dl.load(loadFile)).__class__.__bases__:
            _err = f'\nfile "{loadFile.name}" is not a NeuralNetworkParser'
            raise NeuralNetworkParserError(_err)

        return nn


# todo: implement accuracy history, accuracy plotting, accuracy vs cost plotting. *L
class PlotNeuralNetwork(Plot):
    # plots cost graphs of neural networks
    @staticmethod
    def plotCostGraph(nn: "AbstractNeuralNetwork") -> "None":
        yh = np.array(nn.costHistory)
        yh[0][0] = 0
        xh = np.arange(yh.size).reshape(yh.shape)
        minus = np.ones_like(xh)
        minus[0, :] = 0
        xh = xh - minus
        PlotNeuralNetwork.plotHeight(xh, yh)
        PlotNeuralNetwork.show()
