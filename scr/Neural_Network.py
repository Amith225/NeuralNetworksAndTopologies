import time as tm

import numpy as np
from matplotlib import collections as mc, pyplot as plt


class CreateNeuralNetwork:
    def __init__(self, shape, initializer, activation, output_activation=None):
        self.shape = shape
        self.layers = len(self.shape)
        if output_activation is None: output_activation = activation
        self.weights, self.biases = initializer(self)
        self.activation, self.activated_derivative = activation
        self.output_activation, self.activated_output_derivative = output_activation
        self.activated_outputs = np.array([np.zeros((self.shape[i], 1), dtype=np.float32) for i in range(self.layers)],
                                          dtype=np.ndarray)
        self.delta_weights, self.delta_biases = Initializer.normal(0)(self)

        self.costs = []
        self.cost = 0
        self.e = 0
        self.train_database = None
        self.epochs = None
        self.batch_size = None
        self.loss_function = None
        self.optimizer = None
        self.opt = None
        self.cost_derivative = None

    def process(self, input):
        self.activated_outputs[0] = np.array(input, dtype=np.float32).reshape((len(input), 1))

        for l in range(self.layers - 2):
            self.activated_outputs[l + 1] = self.activation(
                np.einsum('ij,jk->ik', self.weights[l], self.activated_outputs[l],
                          dtype=np.float32) + self.biases[l])

        return self.output_activation(np.einsum('ij,jk->ik', self.weights[l + 1], self.activated_outputs[l + 1],
                                                dtype=np.float32) + self.biases[l])

    def forward_pass(self, input):
        self.activated_outputs[0] = input

        for l in range(self.layers - 2):
            self.activated_outputs[l + 1] = self.activation(
                np.einsum('ij,ljk->lik', self.weights[l], self.activated_outputs[l],
                          dtype=np.float32) + self.biases[l])

        self.activated_outputs[l + 2] = self.output_activation(
            np.einsum('ij,ljk->lik', self.weights[l + 1], self.activated_outputs[l + 1],
                      dtype=np.float32) + self.biases[l + 1])

    def delta_update(self, layer, activated_derivative, theta):
        layer = layer - 1
        delta_biases = self.cost_derivative * activated_derivative(self.activated_outputs[layer + 1])
        self.delta_biases[layer] = delta_biases.mean(axis=0)
        self.delta_weights[layer] = np.einsum('lij,lji->lij', delta_biases, self.activated_outputs[layer],
                                              dtype=np.float32).mean(axis=0)

        self.cost_derivative = np.einsum('ij,lik->ljk', theta, self.cost_derivative, dtype=np.float32)

        self.opt(layer)
        self.weights[layer] -= self.delta_weights[layer]
        self.biases[layer] -= self.delta_biases[layer]

    def train(self, train_database=None, epochs=None, batch_size=None, loss_function=None, optimizer=None):
        if train_database is not None: self.train_database = train_database
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if loss_function is not None: self.loss_function = loss_function
        if optimizer is not None: self.optimizer, self.opt = optimizer

        train_costs = []
        for e in range(self.epochs):
            print('epoch:', e, end='  ')
            self.train_database.set_batch_size(self.batch_size)
            t = tm.time()
            self.cost = 0
            for b in range(self.train_database.shape[0] // self.train_database.batch_size):
                self.optimizer(self.train_database.next_mini_batch())
            cost = self.cost / self.train_database.batch_size
            print('cost:', cost, 'time:', tm.time() - t)
            train_costs.append(cost)

            self.e += e * self.train_database.batch_size / self.train_database.shape[0]
        self.costs.append(train_costs)

        self.activated_outputs = np.array([np.zeros((self.shape[i], 1), dtype=np.float32) for i in range(self.layers)],
                                          dtype=np.ndarray)

    def test(self):
        pass


class Initializer:
    @staticmethod
    def uniform(start, stop):
        def initializer(self):
            weights = [np.random.uniform(start, stop, (self.shape[i], self.shape[i - 1])).astype(dtype=np.float32)
                       for i in range(1, self.layers)]
            biases = [np.random.uniform(start, stop, (self.shape[i], 1)).astype(dtype=np.float32)
                      for i in range(1, self.layers)]

            return weights, biases

        return initializer

    @staticmethod
    def normal(scale=1):
        def initializer(self):
            weights = [(np.random.default_rng().standard_normal((self.shape[i], self.shape[i - 1]),
                                                                dtype=np.float32)) * scale
                       for i in range(1, self.layers)]
            biases = [(np.random.default_rng().standard_normal((self.shape[i], 1),
                                                               dtype=np.float32)) * scale
                      for i in range(1, self.layers)]

            return weights, biases

        return initializer

    @staticmethod
    def xavier(he=1):
        def initializer(self):
            weights = [np.random.default_rng().standard_normal((self.shape[i], self.shape[i - 1]),
                                                               dtype=np.float32) * (he / self.shape[i - 1]) ** 0.5
                       for i in range(1, self.layers)]
            biases = [np.random.default_rng().standard_normal((self.shape[i], 1),
                                                              dtype=np.float32) * (he / self.shape[i - 1]) ** 0.5
                      for i in range(1, self.layers)]

            return weights, biases

        return initializer


class LossFunction:
    @staticmethod
    def mean_square():
        def loss_function(self, b):
            self.forward_pass(b[0])
            cost_derivative = self.activated_outputs[-1] - b[1]

            return cost_derivative, np.einsum('lij,lij->', cost_derivative, cost_derivative, dtype=np.float32)

        return loss_function

    @staticmethod
    def cross_entropy():
        def loss_function(self, b):
            pass

        return loss_function


class ActivationFunction:
    @staticmethod
    def sigmoid(alpha=1, beta=0):
        e = np.float32(np.e)

        def activation(x):
            return 1 / (1 + e ** (-alpha * (x + beta)))

        def activated_derivative(activated_x):
            return alpha * (activated_x * (1 - activated_x))

        return activation, activated_derivative

    @staticmethod
    def relu():
        def activation(x):
            return x * (x > 0)

        def activated_derivative(activated_x):
            return np.float32(1) * (activated_x != 0)

        return activation, activated_derivative

    @staticmethod
    def tanh(alpha=1):
        def activation(x):
            return np.arctan(alpha * x)

        def activated_derivative(activated_x):
            return alpha * np.square(np.cos(activated_x))

        return activation, activated_derivative

    @staticmethod
    def softmax():
        e = np.float32(np.e)

        def activation(x):
            numerator = e ** (x - x.max())

            return numerator / np.einsum('lij->l', numerator.transpose(), dtype=np.float32)

        def activated_derivative(activated_x):
            j = -np.einsum('lij,lkj->lik', activated_x, activated_x)

            d_i = np.diag_indices_from(j[0])
            j[:, d_i[0], d_i[1]] = np.einsum('lij,lij->li', activated_x, (1 - activated_x))

            return j.sum(axis=2, keepdims=1)

        return activation, activated_derivative


class Optimizer:
    @staticmethod
    def traditional_gradient_decent(this, lr):
        def opt(l):
            this.delta_weights[l], this.delta_biases[l] = lr * this.delta_weights[l], lr * this.delta_biases[l]

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1]) for l in
             range(this.layers - 2, 0, -1)]

        return optimizer, opt

    @staticmethod
    def moment(this, lr, alpha=None):
        if alpha is None: alpha = lr
        this.prev_delta_weights, this.prev_delta_biases = Initializer.normal(0)(this)

        def opt(l):
            this.delta_weights[l] = this.prev_delta_weights[l] = alpha * this.prev_delta_weights[l] + lr * \
                                                                 this.delta_weights[l]
            this.delta_biases[l] = this.prev_delta_biases[l] = alpha * this.prev_delta_biases[l] + lr * \
                                                               this.delta_biases[l]

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1]) for l in
             range(this.layers - 2, 0, -1)]

        return optimizer, opt

    @staticmethod
    def decay(this, lr, alpha=None):
        if alpha is None: alpha = lr

        def opt(l):
            k = lr / (1 + this.e / alpha)
            this.delta_weights[l] = k * this.delta_weights[l]
            this.delta_biases[l] = k * this.delta_biases[l]

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1]) for l in
             range(this.layers - 2, 0, -1)]

        return optimizer, opt

    @staticmethod
    def nesterov(this, lr, alpha=None):
        if alpha is None: alpha = lr
        this.prev_delta_weights, this.prev_delta_biases = Initializer.normal(0)(this)

        def opt(l):
            this.delta_weights[l] = this.prev_delta_weights[l] = alpha * this.prev_delta_weights[l] + lr * \
                                                                 this.delta_weights[l]
            this.delta_biases[l] = this.prev_delta_biases[l] = alpha * this.prev_delta_biases[l] + lr * \
                                                               this.delta_biases[l]

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative,
                              this.weights[this.layers - 2] - this.prev_delta_weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1] - this.prev_delta_weights[l - 1])
             for l in range(this.layers - 2, 0, -1)]

        return optimizer, opt

    @staticmethod
    def adagrad(this, lr=0.01, epsilon=np.e ** -8):
        this.gti_w, this.gti_b = Initializer.normal(0)(this)

        def opt(l):
            d_w, d_b = this.delta_weights[l], this.delta_biases[l]
            this.gti_w[l] += np.square(d_w)
            this.gti_b[l] += np.square(d_b)

            this.delta_weights[l] = lr * d_w / np.sqrt(epsilon + this.gti_w[l])
            this.delta_biases[l] = lr * d_b / np.sqrt(epsilon + this.gti_b[l])

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1]) for l in
             range(this.layers - 2, 0, -1)]

        return optimizer, opt

    # Know some what how this work, but not sure
    @staticmethod
    def adadelta(this, lr=1, alpha=0.95, epsilon=np.e ** -16):
        alpha_bar = 1 - alpha
        this.vt_w, this.vt_b = Initializer.normal(0)(this)
        this.wt_w, this.wt_b = Initializer.normal(0)(this)

        def opt(l):
            this.vt_w[l] = alpha * this.vt_w[l] + alpha_bar * np.square(this.delta_weights[l])
            this.vt_b[l] = alpha * this.vt_b[l] + alpha_bar * np.square(this.delta_biases[l])
            this.delta_weights[l] = np.sqrt((epsilon + this.wt_w[l]) / (epsilon + this.vt_w[l])) * this.delta_weights[l]
            this.delta_biases[l] = np.sqrt((epsilon + this.wt_b[l]) / (epsilon + this.vt_b[l])) * this.delta_biases[l]
            this.wt_w[l] = alpha * this.wt_w[l] + alpha_bar * np.square(this.delta_biases[l])
            this.wt_b[l] = alpha * this.wt_b[l] + alpha_bar * np.square(this.delta_biases[l])

            this.delta_weights[l] = lr * this.delta_weights[l]
            this.delta_biases[l] = lr * this.delta_biases[l]

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1]) for l in
             range(this.layers - 2, 0, -1)]

        return optimizer, opt

    # no understanding at all
    @staticmethod
    def adam(this, lr=0.005, alpha=0.9, beta=0.999, epsilon=np.e ** -16):
        alpha_bar = 1 - alpha
        beta_bar = 1 - beta
        this.mt_w, this.mt_b = Initializer.normal(0)(this)
        this.vt_w, this.vt_b = Initializer.normal(0)(this)
        this.t = 0

        def opt(l):
            this.t += 1
            this.mt_w[l] = alpha * this.mt_w[l] + alpha_bar * this.delta_weights[l]
            this.mt_b[l] = alpha * this.mt_b[l] + alpha_bar * this.delta_biases[l]
            this.vt_w[l] = beta * this.vt_w[l] + beta_bar * (this.delta_weights[l] ** 2)
            this.vt_b[l] = beta * this.vt_b[l] + beta_bar * (this.delta_biases[l] ** 2)

            m_dw_corr = this.mt_w[l] / (1 - alpha ** this.t)
            m_db_corr = this.mt_b[l] / (1 - alpha ** this.t)
            v_dw_corr = this.vt_w[l] / (1 - beta ** this.t)
            v_db_corr = this.vt_b[l] / (1 - beta ** this.t)

            this.delta_weights[l] = lr * (m_dw_corr / (np.sqrt(v_dw_corr) + epsilon))
            this.delta_biases[l] = lr * (m_db_corr / (np.sqrt(v_db_corr) + epsilon))

        def optimizer(b):
            this.cost_derivative, cost = this.loss_function(this, b)
            this.cost += cost

            this.delta_update(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.delta_update(l, this.activated_derivative, this.weights[l - 1]) for l in
             range(this.layers - 2, 0, -1)]

        return optimizer, opt


class CreateDatabase:
    def __init__(self, input_data, labels):
        self.input_data = np.array(list(input_data), dtype=np.float32).reshape((len(input_data), len(input_data[0]), 1))
        self.labels = np.array(list(labels), dtype=np.float32).reshape((len(labels), len(labels[0]), 1))

        self.shape = self.input_data.shape

        self.batch_size = -1
        self.mini_batch_i = 0

    def set_batch_size(self, batch_size=-1):
        self.batch_size = -1
        self.mini_batch_i = 0

        if batch_size < 0:
            self.batch_size = self.shape[0]
        else:
            self.batch_size = batch_size

        batch_i = np.random.choice(self.shape[0], self.shape[0], replace=False)
        self.input_data, self.labels = self.input_data[batch_i], self.labels[batch_i]


    def next_mini_batch(self):
        start = self.mini_batch_i * self.batch_size
        if start + self.batch_size >= self.shape[0]:
            stop = self.shape[0]
        else:
            stop = start + self.batch_size

        self.mini_batch_i += 1

        return self.input_data[start: stop], self.labels[start: stop]


class LoadNeuralNetwork:
    pass


class SaveNeuralNetwork:
    pass


class PlotGraph:
    @staticmethod
    def plot_cost_graph(nn):
        costs = []
        i = 0
        for cs in nn.costs:
            costs.append([(c + i, j) for c, j in enumerate(cs)])
            i += len(cs) - 1

        lc = mc.LineCollection(costs, colors=['red', 'green'], linewidths=1)
        sp = plt.subplot()
        sp.add_collection(lc)

        sp.autoscale()
        sp.margins(0.1)
        plt.show()
