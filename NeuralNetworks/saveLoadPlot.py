# library direct imports
# library imports
import os
import random as rd
from typing import *

import dill
# library from imports
from matplotlib import collections as mc, pyplot as plt

# library imports for type checking
if TYPE_CHECKING:
    from .neuralNetwork import *

# setup for plotting
colDict = mc.mcolors.cnames
colDict.pop('black')
colors = list(mc.mcolors.cnames.values())
plt.style.use('dark_background')


class SaveNeuralNetwork:  # save neural network class
    # saves neural network as .nns file
    @staticmethod
    def save(this: "AbstractNeuralNetwork", fname: "str" = None) -> Union["str", "None"]:
        if fname is None:
            fname = 'nn'
        if len(fname) >= 4 and '.nns' == fname[-4:0]:
            fname.replace('.nns', '')
        try:
            cost = str(round(this.costs[-1][-1] * 100, 2))
        except IndexError:
            if input("trying to save untrained model, do you want to continue?(y,n): ").lower() != 'y':
                return
            cost = ''
        epoch = sum([len(c) for c in this.costs])
        fname += 'c' + cost + 'e' + str(epoch) + 't' + str(round(this.timeTrained / 60, 2)) + \
                 str(this.wbShape).replace('(', 's').replace(')', '').replace(' ', '')

        trainDatabase, outputs, target = this.trainDatabase, this.wbOutputs, this.target
        loss, deltaLoss = this.loss, this.deltaLoss
        this.trainDatabase = this.wbOutputs = this.target = this.loss = this.deltaLoss = None
        fpath = os.getcwd() + '\\Models\\'
        spath = fpath + fname + '.nns'
        os.makedirs(fpath, exist_ok=True)

        i = 0
        nSpath = spath
        while 1:
            if i != 0:
                nSpath = spath + ' (' + str(i) + ')'
            if os.path.exists(nSpath):
                i += 1
            else:
                spath = nSpath
                break
        dill.dump(this, open(spath, 'wb'))
        this.trainDatabase, this.wbOutputs, this.target = trainDatabase, outputs, target
        this.loss, this.deltaLoss = loss, deltaLoss

        return spath


class LoadNeuralNetwork:  # load neural network class
    # imports neural network from a file
    @staticmethod
    def load(file: "str") -> "AbstractNeuralNetwork":
        if file:
            if not os.path.dirname(file):
                file = os.getcwd() + '\\Models\\' + file
            elif '\\' == file[0] or '/' == file[0] or '//' == file[0]:
                file = os.getcwd() + file

        return dill.load(open(file, 'rb'))


class PlotNeuralNetwork:  # plot neural network class
    # plots cost graphs of neural networks
    @staticmethod
    def plotCostGraph(nn: "AbstractNeuralNetwork") -> "None":
        costs = []
        i = 0
        for costIndex in range(len(nn.costs)):
            cost = nn.costs[costIndex]
            if costIndex > 0:
                costs.append([costs[-1][-1], (i, cost[0])])
            costs.append([(c + i, j) for c, j in enumerate(cost)])
            i += len(cost)

        rd.shuffle(colors)
        lc = mc.LineCollection(costs, colors=colors, linewidths=1, antialiaseds=True)
        sp = plt.subplot()
        sp.add_collection(lc)

        sp.autoscale()
        sp.margins(0.1)
        plt.show()
