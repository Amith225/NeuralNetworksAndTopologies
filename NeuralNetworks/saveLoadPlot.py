import os as _os
import typing as _tp
import random as _rd

import dill as _dl
from matplotlib import collections as _mc, pyplot as _plt

# library imports for type checking
if _tp.TYPE_CHECKING:
    from NeuralNetworks import AbstractNeuralNetwork

# setup for plotting
colDict = _mc.mcolors.cnames
colDict.pop('black')
colors = list(_mc.mcolors.cnames.values())
_plt.style.use('dark_background')


class SaveNeuralNetwork:  # save neural network class
    # saves neural network as .nns file
    @staticmethod
    def save(this: "AbstractNeuralNetwork", fname: "str" = None) -> _tp.Union["str", "None"]:
        if fname is None:
            fname = 'nn'
        if len(fname) >= 4 and '.nns' == fname[-4:0]:
            fname.replace('.nns', '')
        try:
            cost = str(round(this.costHistory[-1][-1] * 100, 2))
        except IndexError:
            if input("trying to save untrained model, do you want to continue?(y,n): ").lower() != 'y':
                return
            cost = ''
        epoch = sum([len(c) for c in this.costHistory])
        fname += 'c' + cost + 'e' + str(epoch) + 't' + str(round(this.timeTrained / 60, 2))

        trainDatabase = this.trainDatabase
        this.trainDatabase = None
        fpath = _os.getcwd() + '\\Models\\'
        spath = fpath + fname + '.nns'
        _os.makedirs(fpath, exist_ok=True)

        i = 0
        nSpath = spath
        while 1:
            if i != 0:
                nSpath = spath + ' (' + str(i) + ')'
            if _os.path.exists(nSpath):
                i += 1
            else:
                spath = nSpath
                break
        _dl.dump(this, open(spath, 'wb'))
        this.trainDatabase = trainDatabase

        return spath


class LoadNeuralNetwork:  # load neural network class
    # imports neural network from a file
    @staticmethod
    def load(file: "str") -> "AbstractNeuralNetwork":
        if file:
            if not _os.path.dirname(file):
                file = _os.getcwd() + '\\Models\\' + file
            elif '\\' == file[0] or '/' == file[0] or '//' == file[0]:
                file = _os.getcwd() + file

        return _dl.load(open(file, 'rb'))


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
