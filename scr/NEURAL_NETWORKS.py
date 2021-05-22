import os
import time as tm
import warnings
from typing import *

import dill
import numpy as np
from matplotlib import collections as mc, pyplot as plt

import TOPOLOGIES as Tp
import print_vars as pv

np.NONE = [np.array([None])]
warnings.filterwarnings('ignore')


class CreateNeuralNetwork:
    def __init__(self,
                 shape: Tuple[int, ...] = (1, 1),
                 initializer: Tp.Initializer = Tp.Initializer.normal(),
                 activation_function: Tp.ActivationFunction = Tp.ActivationFunction.relu(),
                 output_activation_function: Tp.ActivationFunction = Tp.ActivationFunction.softmax()):
        self.shape = tuple(shape)
        self.initializer = initializer
        self.activation, self.activated_derivative = activation_function.activations
        self.output_activation, self.output_activated_derivative = output_activation_function.activations

        self.layers = len(self.shape)
        self.biases, self.weights = self.initializer.initialize(self.shape, self.layers)
        self.theta = self.weights.copy()
        self.biases_ones = np.NONE + [np.ones_like(bias, dtype=np.float32) for bias in self.biases[1:]]

        self.delta_weights, self.delta_biases = None, None
        self.train_database = None
        self.epochs = None
        self.epoch = None
        self.batch = None
        self.batch_size = None
        self.bs = None
        self.batch_length = None
        self.loss_function = None
        self.optimizer = None
        self.outputs = None
        self.targets = None
        self.errors = None
        self.loss = None
        self.loss_derivative = None
        self.costs = []

    # recursive pass
    def forward_pass(self, layer=1):
        if layer == self.layers - 1:
            self.fire(layer, self.output_activation)
        else:
            self.fire(layer, self.activation)
            self.forward_pass(layer + 1)

    def process(self, inputs, layer=1):
        self.outputs[layer - 1] = np.array(inputs, dtype=np.float32)
        self.forward_pass(layer)

        return self.outputs[-1]

    def fire(self, layer, activation):
        self.outputs[layer] = \
            activation(np.einsum('lkj,ik->lij', self.outputs[layer - 1], self.weights[layer]) +
                       self.biases[layer])

    def wire(self, layer):
        # optimization to sum on column(next line only), 5% time reduction
        self.biases[layer] -= (self.delta_biases[layer] * self.biases_ones[layer])[0]
        self.weights[layer] -= self.delta_weights[layer]
        self.theta = self.weights.copy()

    # recursive propagation
    def back_propagation(self, activated_derivative, layer=-1):
        if layer <= -self.layers:
            return
        np.einsum('lij,lim->lij', self.loss_derivative, activated_derivative(self.outputs[layer]),
                  out=self.delta_biases[layer])
        np.einsum('lkj,lij->ik', self.outputs[layer - 1], self.delta_biases[layer], out=self.delta_weights[layer])
        self.loss_derivative = np.einsum('lij,ik->lkj', self.loss_derivative, self.theta[layer])
        self.optimizer.optimize(layer)
        self.wire(layer)
        self.back_propagation(self.activated_derivative, layer - 1)

    def trainer(self,
                train_database: Tp.DataBase = None,
                loss_function: Tp.LossFunction = None,
                optimizer: Tp.Optimizer = None,
                epochs: int = None,
                batch_size: int = None):
        if train_database is not None:
            self.train_database = train_database
        if loss_function is not None:
            self.loss_function = loss_function
        if optimizer is not None:
            self.optimizer = optimizer
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size < 0:
            self.bs = self.train_database.size - batch_size - 1
        else:
            self.bs = self.batch_size

        self.outputs = [np.zeros((self.bs, self.shape[layer], 1), dtype=np.float32)
                        for layer in range(self.layers)]
        self.targets = self.outputs[-1].copy()
        self.errors = self.targets.copy()

        self.delta_biases, self.delta_weights = self.delta_initializer()

    def delta_initializer(self, bs=None):
        if bs is None:
            bs = self.bs
        delta_biases = np.NONE + [(np.zeros((bs, self.shape[i], 1), dtype=np.float32))
                                  for i in range(1, self.layers)]
        delta_weights = Tp.Initializer.normal(0).initialize(self.shape, self.layers)[1]

        return delta_biases, delta_weights

    def train(self):
        costs = [0]
        tot_time = 0
        self.batch_length = int(np.ceil(self.train_database.size / self.bs))
        for self.epoch in range(self.epochs):
            batch_generator = self.train_database.batch_generator(self.bs)
            cost = 0
            time = tm.time()
            for self.batch in range(self.batch_length):
                self.outputs[0], self.targets = batch_generator.__next__()
                self.forward_pass()
                self.errors = self.outputs[-1] - self.targets
                self.loss, self.loss_derivative = self.loss_function.loss_function(self.errors)
                self.back_propagation(self.output_activated_derivative)
                cost += self.loss
            time = tm.time() - time
            cost /= self.train_database.size
            costs.append(cost)
            tot_time += time
            print(end='\r')
            print(pv.CBOLD + pv.CBLUE + pv.CURL + f'epoch:{self.epoch}' + pv.CEND,
                  pv.CYELLOW + f'cost:{cost}', f'time:{time}' + pv.CEND,
                  pv.CBOLD + f'eta:{tot_time / (self.epoch + 1) * (self.epochs - self.epoch - 1)}' + pv.CEND, end='')
        print()
        print(pv.CBOLD + pv.CRED2 + f'tot_time:{tot_time}', f'avg_time:{tot_time / self.epochs}' + pv.CEND)
        self.costs.append(costs[1:])


class PlotNeuralNetwork:
    @staticmethod
    def plot_cost_graph(nn):
        costs = []
        i = 0
        for cost_i in range(len(nn.costs)):
            cost = nn.costs[cost_i]
            if cost_i > 0:
                costs.append([costs[-1][-1], (i, cost[0])])
            costs.append([(c + i, j) for c, j in enumerate(cost)])
            i += len(cost)

        lc = mc.LineCollection(costs, colors=['red', 'red', 'green', 'green'], linewidths=1)
        sp = plt.subplot()
        sp.add_collection(lc)

        sp.autoscale()
        sp.margins(0.1)
        plt.show()


class SaveNeuralNetwork:
    @staticmethod
    def save(this, fname='nn'):
        if len(fname) >= 4 and '.nns' == fname[-4:0]:
            fname.replace('.nns', '')
        cost = str(round(this.costs[-1][-1] * 100, 2))
        cost.replace('.', ':')
        fname += 'c' + cost
        train_database = this.train_database
        this.train_database = None
        dill.dump(this, open(os.path.dirname(os.getcwd()) + '\\models\\' + fname + '.nns', 'wb'))
        this.train_database = train_database


class LoadNeuralNetwork:
    @staticmethod
    def load(fname, fpath=None):
        if fpath is None:
            return dill.load(open(os.path.dirname(os.getcwd()) + '\\models\\' + fname, 'rb'))
        else:
            return dill.load(open(fpath, 'rb'))
