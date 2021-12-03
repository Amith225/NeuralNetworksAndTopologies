import typing as _tp
import random as _rd

import dill as _dl
import numpy as _np
from matplotlib import collections as _mc, pyplot as _plt, widgets as _wg

from utils import AbstractLoad, Plot
from NeuralNetworks.neuralNetwork import AbstractNeuralNetwork
from Topologies.dataBase import DataBase

# library imports for type checking
if _tp.TYPE_CHECKING:
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
        if AbstractNeuralNetwork not in (nn := _dl.load(loadFile)).__class__.__bases__:
            _err = f'\nfile "{loadFile.name}" is not a NeuralNetworkParser'
            raise NeuralNetworkParserError(_err)

        return nn


class PlotNeuralNetwork(Plot):
    # plots cost graphs of neural networks
    @staticmethod
    def plotCostGraph(nn: "AbstractNeuralNetwork") -> "None":
        yh = _np.array(nn.costHistory)
        yh[0][0] = 0
        xh = _np.arange(yh.size).reshape(yh.shape)
        minus = _np.ones_like(xh)
        minus[0, :] = 0
        xh = xh - minus
        PlotNeuralNetwork.plot(xh, yh)
        PlotNeuralNetwork.show()

    @staticmethod
    def plotInputVecAsImg(inpVec: "_np.ndarray"):
        shape = inpVec.shape[1:]
        rows, columns = 5, 5
        MAX = rows * columns
        figShapeFactor = max(shape)
        fig = _plt.figure(figsize=(shape[0] / figShapeFactor * 7, shape[1] / figShapeFactor * 7))
        butPrev = _wg.Button(_plt.axes([0, 0, .5, .05]), '<-', color='red', hovercolor='blue')
        butNext = _wg.Button(_plt.axes([.5, 0, .5, .05]), '->', color='red', hovercolor='blue')
        axes = [fig.add_subplot(rows, columns, i + 1) for i in range(MAX)]
        fig.page = 0

        def onclick(_page):
            if _page == 0:
                butPrev.active = False
                butPrev.ax.patch.set_visible(False)
            else:
                butPrev.active = True
                butPrev.ax.patch.set_visible(True)
            if _page == (inpVec.shape[0] - 1) // MAX:
                butNext.active = False
                butNext.ax.patch.set_visible(False)
            else:
                butNext.active = True
                butNext.ax.patch.set_visible(True)
            fig.page = _page
            to = (_page + 1) * MAX
            if to > inpVec.shape[0]:
                to = inpVec.shape[0]
            [(ax.clear(),) for ax in axes]
            for i, im in enumerate(inpVec[_page * MAX:to]):
                axes[i].imshow(im)
                # axes[i].text(-1, 3, i + MAX * _page)
                axes[i].set_yticklabels([])
                axes[i].set_xticklabels([])
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.canvas.draw()
            fig.canvas.flush_events()
        butNext.on_clicked(lambda *_: onclick(fig.page + 1))
        butPrev.on_clicked(lambda *_: onclick(fig.page - 1))
        onclick(fig.page)

        _plt.show()
