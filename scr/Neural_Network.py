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
        self.training_set = None
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

        return self.activation(np.einsum('ij,jk->ik', self.weights[l], self.activated_outputs[l],
                                         dtype=np.float32) + self.biases[l])

    def forward_pass(self, input):
        self.activated_outputs[0] = input

        for l in range(self.layers - 2):
            self.activated_outputs[l + 1] = self.activation(
                np.einsum('ij,jk->ik', self.weights[l], self.activated_outputs[l],
                          dtype=np.float32) + self.biases[l])

        self.activated_outputs[l + 2] = self.output_activation(
            np.einsum('ij,jk->ik', self.weights[l + 1], self.activated_outputs[l + 1],
                      dtype=np.float32) + self.biases[l + 1])

    def back_propagation(self, b):
        self.cost_derivative, cost = self.loss_function(self, b)
        self.cost += cost

        self.optimizer()

    def find_delta(self, layer, activated_derivative, theta):
        layer = layer - 1
        delta_biases = self.cost_derivative * activated_derivative(self.activated_outputs[layer + 1])
        np.einsum('ij,ji->ij', delta_biases, self.activated_outputs[layer], dtype=np.float32,
                  out=self.delta_weights[layer])

        self.cost_derivative = np.einsum('ij,ik', theta, self.cost_derivative, dtype=np.float32)

        self.opt(layer)
        self.weights[layer] -= self.delta_weights[layer]
        self.biases[layer] -= self.delta_biases[layer]

    def train(self, training_set=None, epochs=None, batch_size=None, loss_function=None, optimizer=None,
              vectorize=True):
        if vectorize is True and training_set is not None:
            training_set = np.array([[np.array(t[0], dtype=np.float32).reshape((len(t[0]), 1)),
                                      np.array(t[1], dtype=np.float32).reshape((len(t[1]), 1))]
                                     for t in training_set], dtype=np.ndarray)
        if training_set is not None: self.training_set = training_set
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if loss_function is not None: self.loss_function = loss_function
        if optimizer is not None: self.optimizer, self.opt = optimizer

        if self.batch_size < 0:
            batch_size = len(self.training_set) + self.batch_size
        else:
            batch_size = self.batch_size

        train_costs = []
        for e in range(self.epochs):
            print('epoch:', e, end='  ')
            batch_set = self.training_set[np.random.choice(self.training_set.shape[0], batch_size, replace=False)]

            t = tm.time()
            self.cost = 0
            for b in batch_set:
                self.back_propagation(b)
            cost = self.cost / batch_size
            print('cost:', cost, 'time:', tm.time() - t)
            train_costs.append(cost)

            self.e += e * batch_size / len(self.training_set)
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

            return np.array(weights, dtype=np.ndarray), np.array(biases, dtype=np.ndarray)

        return initializer

    @staticmethod
    def normal(scale=1):
        def initializer(self):
            weights = [np.random.default_rng().standard_normal((self.shape[i], self.shape[i - 1]),
                                                               dtype=np.float32)
                       for i in range(1, self.layers)]
            biases = [np.random.default_rng().standard_normal((self.shape[i], 1),
                                                              dtype=np.float32)
                      for i in range(1, self.layers)]

            return np.array(weights, dtype=np.ndarray) * scale, np.array(biases, dtype=np.ndarray) * scale

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

            return np.array(weights, dtype=np.ndarray), np.array(biases, dtype=np.ndarray)

        return initializer


class LossFunction:
    @staticmethod
    def mean_square():
        def loss_function(self, b):
            self.forward_pass(b[0])
            cost_derivative = self.activated_outputs[-1] - b[1]

            return cost_derivative, np.einsum('ij,ij->', cost_derivative, cost_derivative, dtype=np.float32)

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

            return numerator / np.einsum('ij->', numerator, dtype=np.float32)

        def activated_derivative(activated_x):
            j = -np.einsum('ij,kj', activated_x, activated_x)
            j[np.diag_indices_from(j)] = np.einsum('ij,ij->ji', activated_x, (1 - activated_x))

            return j.sum(axis=1, keepdims=1)

        return activation, activated_derivative


class Optimizer:
    @staticmethod
    def traditional_gradient_decent(this, lr):
        def opt(l):
            this.delta_weights[l], this.delta_biases[l] = lr * this.delta_weights[l], lr * this.delta_biases[l]

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1]) for l in range(this.layers - 2, 0, -1)]

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

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1]) for l in range(this.layers - 2, 0, -1)]

        return optimizer, opt

    @staticmethod
    def decay(this, lr, alpha=None):
        if alpha is None: alpha = lr

        def opt(l):
            k = lr / (1 + this.e / alpha)
            this.delta_weights[l] = k * this.delta_weights[l]
            this.delta_biases[l] = k * this.delta_biases[l]

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1]) for l in range(this.layers - 2, 0, -1)]

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

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative,
                            this.weights[this.layers - 2] - this.prev_delta_weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1] - this.prev_delta_weights[l - 1])
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

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1]) for l in range(this.layers - 2, 0, -1)]

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

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1]) for l in range(this.layers - 2, 0, -1)]

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

        def optimizer():
            this.find_delta(this.layers - 1, this.activated_output_derivative, this.weights[this.layers - 2])
            [this.find_delta(l, this.activated_derivative, this.weights[l - 1]) for l in range(this.layers - 2, 0, -1)]

        return optimizer, opt


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
