import time as tm

import numpy as np

import Topology as Tp


class CreateNeuralNetwork:
    def __init__(self, shape=None, initializer=None, activation=None, output_activation=None):
        if shape is None:
            shape = (1, 1)
        if initializer is None:
            initializer = Tp.Initializer.xavier(2)
        if activation is None:
            activation = Tp.ActivationFunction.relu()
        if output_activation is None:
            output_activation = Tp.ActivationFunction.softmax()
        self.shape = shape
        self.layers = len(self.shape)
        self.weights, self.biases = initializer(self)
        self.activation, self.activated_derivative = activation
        self.output_activation, self.activated_output_derivative = output_activation
        self.activated_outputs = np.array([np.zeros((self.shape[i], 1), dtype=np.float32) for i in range(self.layers)],
                                          dtype=np.ndarray)
        self.delta_weights, self.delta_biases = Tp.Initializer.normal(0)(self)

        self.costs = []
        self.cost = 0
        self.e = 0

        self.train_database = None
        self.epochs = 1
        self.batch_size = -1
        self.loss_function = Tp.LossFunction.mean_square()
        self.optimizer, self.opt = Tp.Optimizer.adadelta(self)
        self.cost_derivative = None
        self.l = self.layers - 3

    def process(self, inp):
        self.activated_outputs[0] = np.array(inp, dtype=np.float32).reshape((len(inp), 1))

        for self.l in range(self.layers - 2):
            self.activated_outputs[self.l + 1] = self.activation(
                np.einsum('ij,jk->ik', self.weights[self.l], self.activated_outputs[self.l],
                          dtype=np.float32) + self.biases[self.l])

        return self.output_activation(np.einsum('ij,jk->ik', self.weights[self.l + 1],
                                                self.activated_outputs[self.l + 1],
                                                dtype=np.float32) + self.biases[self.l + 1])

    def forward_pass(self, inp):
        self.activated_outputs[0] = inp

        for self.l in range(self.layers - 2):
            self.activated_outputs[self.l + 1] = self.activation(
                np.einsum('ij,jk->ik', self.weights[self.l], self.activated_outputs[self.l],
                          dtype=np.float32) + self.biases[self.l])

        self.activated_outputs[self.l + 2] = self.output_activation(
            np.einsum('ij,jk->ik', self.weights[self.l + 1], self.activated_outputs[self.l + 1],
                      dtype=np.float32) + self.biases[self.l + 1])

    def delta_update(self, layer, activated_derivative, theta):
        layer = layer - 1
        self.delta_biases[layer] = self.cost_derivative * activated_derivative(self.activated_outputs[layer + 1])
        np.einsum('ij,ji->ij', self.delta_biases[layer], self.activated_outputs[layer], dtype=np.float32,
                  out=self.delta_weights[layer])

        self.cost_derivative = np.einsum('ij,ik', theta, self.cost_derivative, dtype=np.float32)

        self.opt(layer)
        self.weights[layer] -= self.delta_weights[layer]
        self.biases[layer] -= self.delta_biases[layer]

    def train(self, train_database=None, epochs=None, batch_size=None, loss_function=None, optimizer=None):
        if train_database is not None:
            self.train_database = train_database
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if loss_function is not None:
            self.loss_function = loss_function
        if optimizer is not None:
            self.optimizer, self.opt = optimizer

        train_costs = []
        for e in range(self.epochs):
            print('epoch:', e, end='  ')
            batch_set = self.train_database.mini_batch(self.batch_size)
            t = tm.time()
            self.cost = 0
            for b in zip(*batch_set):
                self.optimizer(b)
            cost = self.cost / self.train_database.batch_size
            print('cost:', cost, 'time:', tm.time() - t)
            train_costs.append(cost)

            self.e += e * self.train_database.batch_size / self.train_database.shape[0]
        self.costs.append(train_costs)

        self.activated_outputs = np.array([np.zeros((self.shape[i], 1), dtype=np.float32) for i in range(self.layers)],
                                          dtype=np.ndarray)

    def test(self):
        pass


class LoadNeuralNetwork:
    pass


class SaveNeuralNetwork:
    pass
