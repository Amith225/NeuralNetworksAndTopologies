import typing as tp
if tp.TYPE_CHECKING:
    from . import *
    from ..Topologies import *
    from ..Utils import *

import dill as dl

from Utils import AbstractLoad, Plot
from NeuralNetworks.neuralNetwork import AbstractNeuralNetwork


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


class PlotNeuralNetwork(Plot):
    COST_LABEL = 'Cost'
    ACCURACY_LABEL = 'Accuracy(%)'
    EPOCH_LABEL = 'Epoch'

    @staticmethod
    def epochGraphs(yh, yLabel, ylim=None):
        xh = []
        index = 0
        [(xh.append([i + index for i in range(len(y))]), index := index + len(y) - 1) for y in yh]
        ax = PlotNeuralNetwork.plotHeight(xh, yh, cluster=True, scatterLabels=[[f"{yi:.2f}" for yi in y] for y in yh])
        ax.set_ylabel(yLabel)
        ax.set_xlabel(PlotNeuralNetwork.EPOCH_LABEL)
        if ylim is not None:
            ax.set_ylim(ylim)

        return ax

    # plots cost graphs of neural networks
    @staticmethod
    def showCostGraph(nn: "AbstractNeuralNetwork") -> "None":
        yh = nn.costHistory
        yh[0][0] = yh[0][1]
        PlotNeuralNetwork.epochGraphs(yh, PlotNeuralNetwork.COST_LABEL)
        PlotNeuralNetwork.show()

    @staticmethod
    def showAccuracyGraph(nn: "AbstractNeuralNetwork") -> "None":
        yh = nn.accuracyHistory
        PlotNeuralNetwork.epochGraphs(yh, PlotNeuralNetwork.ACCURACY_LABEL, (0, 100))
        PlotNeuralNetwork.show()

    @staticmethod
    def showCostVsAccuracyGraph(nn):
        yh = nn.accuracyHistory
        xh = nn.costHistory
        xh[0][0] = xh[0][1]
        ax = PlotNeuralNetwork.plotHeight(xh, yh, cluster=True, scatterRotation=60,
                                          scatterLabels=[[f"({xi:.2f},{yi:.2f})" for yi, xi in zip(x, y)]
                                                         for x, y in zip(xh, yh)])
        ax.invert_xaxis()
        ax.set_ylim(0, 100)
        ax.set_ylabel(PlotNeuralNetwork.ACCURACY_LABEL)
        ax.set_xlabel(PlotNeuralNetwork.COST_LABEL)
        PlotNeuralNetwork.show()
