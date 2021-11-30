import typing as _tp
import random as _rd

import dill as _dl
from matplotlib import collections as _mc, pyplot as _plt

from utils import AbstractLoad
from NeuralNetworks.neuralNetwork import AbstractNeuralNetwork

# library imports for type checking
if _tp.TYPE_CHECKING:
    pass

# setup for plotting
colDict = _mc.mcolors.cnames
colDict.pop('black')
colors = list(_mc.mcolors.cnames.values())
_plt.style.use('dark_background')


class NeuralNetworkParserError(Exception):
    pass


class LoadNeuralNetwork(AbstractLoad):
    DEFAULT_DIR = AbstractNeuralNetwork.DEFAULT_DIR
    FILE_TYPE = AbstractNeuralNetwork.FILE_TYPE

    def __new__(cls, file, *args, **kwargs) -> "AbstractNeuralNetwork":
        return cls.load(file, *args, **kwargs)

    @classmethod
    def _read(cls, loadFile, *args, **kwargs) -> "AbstractNeuralNetwork":
        if AbstractNeuralNetwork not in (nn := _dl.load(loadFile)).__class__.__bases__:
            _err = f'\nfile "{loadFile.name}" is not a NeuralNetworkParser'
            raise NeuralNetworkParserError(_err)

        return nn


class PlotNeuralNetwork:  # plot neural network class
    # plots cost graphs of neural networks
    @staticmethod
    def plotCostGraph(nn: "AbstractNeuralNetwork") -> "None":
        costs = []
        i = 0
        for costIndex in range(len(nn.costHistory)):
            cost = nn.costHistory[costIndex]
            if costIndex > 0:
                costs.append([costs[-1][-1], (i, cost[0])])
            costs.append([(c + i, j) for c, j in enumerate(cost)])
            i += len(cost)

        _rd.shuffle(colors)
        lc = _mc.LineCollection(costs, colors=colors, linewidths=1, antialiaseds=True)
        sp = _plt.subplot()
        sp.add_collection(lc)

        sp.autoscale()
        sp.margins(0.1)
        _plt.show()
