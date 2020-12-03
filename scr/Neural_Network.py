import numpy as np
import time as tm


class CreateNeuralNetwork:
    def __init__(self, shape, initializer, activation, activation_output=None):
        self.shape = shape
        self.layers = len(self.shape)
        if activation_output is None: activation_output = activation
        self.weights, self.biases = initializer(self)
        self.activation, self.activated_derivative = activation
        self.output_activation, self.activated_output_derivative = activation_output
        self.activated_outputs = [i for i in range(self.layers)]
        self.delta_weights, self.delta_biases = Initializer.normal(0)(self)

        self.costs = []
        self.t = 0
        self.training_set = None
        self.epochs = None
        self.batch_size = None
        self.loss_function = None
        self.optimizer = None
        self.cost_derivative = None
        self.cost = 0

    def process(self, input):
        input = np.array(input, dtype=np.float32).reshape((len(input), 1))

        for l in range(self.layers - 2):
            input = self.activation(self.weights[l] @ input + self.biases[l])

        return self.output_activation(self.weights[l + 1] @ input + self.biases[l + 1])

    def forward_pass(self, input):
        self.activated_outputs[0] = input

        for l in range(self.layers - 2):
            input = self.activation(self.weights[l] @ input + self.biases[l])
            self.activated_outputs[l + 1] = input

        self.activated_outputs[l + 2] = self.output_activation(self.weights[l + 1] @ input + self.biases[l + 1])

    def back_propagation(self, b):
        self.cost_derivative, cost = self.loss_function(self, b)
        self.cost += cost

        self.find_delta(self.layers - 1, self.activated_output_derivative)
        for l in range(self.layers - 2, 0, -1):
            self.find_delta(l, self.activated_derivative)

        self.optimizer()

    def find_delta(self, layer, activated_derivative):
        delta_biases = self.cost_derivative * activated_derivative(self.activated_outputs[layer])
        delta_weights = delta_biases @ self.activated_outputs[layer - 1].transpose()
        self.delta_biases[layer - 1], self.delta_weights[layer - 1] = delta_biases, delta_weights

        self.cost_derivative = self.weights[layer - 1].transpose() @ self.cost_derivative


    def train(self, training_set=None, epochs=None, batch_size=None, loss_function=None, optimizer=None,
              vectorize=True):
        if vectorize is True and training_set is not None:
            training_set = np.array([[np.array(t[0], dtype=np.float32).reshape((len(t[0]), 1)),
                                      np.array(t[1], dtype=np.float32).reshape((len(t[1]), 1))]
                                     for t in training_set], dtype=np.object)
        if training_set is not None: self.training_set = training_set
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if loss_function is not None: self.loss_function = loss_function
        if optimizer is not None: self.optimizer = optimizer

        if self.batch_size < 0: batch_size = len(self.training_set) + self.batch_size
        else: batch_size = self.batch_size

        train_costs = []
        for e in range(self.epochs):
            print('epoch:', e, end='  ')
            batch_set = self.training_set[np.random.choice(self.training_set.shape[0], batch_size, replace=False)]

            t = tm.time()
            self.cost = 0
            for b in batch_set:
                self.back_propagation(b)
            cost = self.cost / batch_size
            print('cost:', cost, 'time:', tm.time()-t)
            train_costs.append(cost)
        self.costs.append([self.t, train_costs])

        self.t += 1
        self.activated_outputs = [i for i in range(self.layers)]

    def quick_train(self):
        pass

    def test(self):
        pass



class Initializer:
    @staticmethod
    def normal(scale=1):
        def initializer(self):
            weights = [np.random.default_rng().standard_normal((self.shape[i], self.shape[i - 1]),
                                                               dtype=np.float32) * scale
                       for i in range(1, self.layers)]
            biases = [np.random.default_rng().standard_normal((self.shape[i], 1),
                                                              dtype=np.float32) for i in range(1, self.layers)]

            return np.array(weights, dtype=np.object), np.array(biases, dtype=np.object)

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

            return np.array(weights, dtype=np.object), np.array(biases, dtype=np.object)

        return initializer



class LossFunction:
    @staticmethod
    def mean_square():
        def loss_function(self, b):
            self.forward_pass(b[0])
            cost_derivative = self.activated_outputs[-1] - b[1]
            cost = cost_derivative ** 2

            return cost_derivative, cost.sum()

        return loss_function



class ActivationFunction:
    @staticmethod
    def sigmoid(alpha=1, beta=0):
        def activation(x):

            return 1 / (1 + np.e ** (-alpha * (x + beta)))

        def activated_derivative(activated_x):

            return alpha * (activated_x * (1 - activated_x))

        return activation, activated_derivative

    @staticmethod
    def relu():  # can make faster in where
        def activation(x):

            return np.where(x < 0, 0, x)

        def activated_derivative(activated_x):

            return np.where(activated_x == 0, 0, 1)

        return activation, activated_derivative

    @staticmethod
    def tanh(alpha=1):
        def activation(x):

            return np.arctan(alpha * x)

        def activated_derivative(activated_x):

            return 1 / (1 + np.tan(activated_x) ** 2)

        return activation, activated_derivative



class Optimizer:
    @staticmethod
    def learning_rate(this, lr):
        def optimizer():

            this.weights -= lr * this.delta_weights
            this.biases -= lr * this.delta_biases

        return optimizer

    @staticmethod
    def moment(this, lr, alpha):
        this.prev_delta_weights, this.prev_delta_biases = Initializer.normal(0)(this)

        def optimizer():
            this.delta_weights[:] = alpha * this.prev_delta_weights + lr * this.delta_weights
            this.delta_biases[:] = alpha * this.prev_delta_biases + lr * this.delta_biases

            this.prev_delta_weights[:] = this.delta_weights
            this.prev_delta_biases[:] = this.delta_biases

            this.weights -= this.delta_weights
            this.biases -= this.delta_biases

        return optimizer



class LoadNeuralNetwork:
    pass



class SaveNeuralNetwork:
    pass



class PlotGraph:
    def plot_cost_graph(self, nn):
        pass
