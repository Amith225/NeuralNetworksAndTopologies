import numpy as np
import time as tm


class CreateNeuralNetwork:
    def __init__(self, shape, initializer, activation, activation_output=None):
        self.shape = shape
        self.layers = len(self.shape)
        if activation_output is None: activation_output = activation

        self.weights, self.biases = initializer(self)
        self.activation, self.activated_derivative = activation
        self.activation_output, self.activated_output_derivative = activation_output
        self.costs = []
        self.t = 0

        self.activated_outputs = [i for i in range(self.layers)]
        self.delta_weights, self.delta_biases = initializer(self, 0, 0)
        self.training_set = None
        self.epochs = None
        self.batch_size = None
        self.loss_function = None
        self.optimizer = None

    def process(self, input):
        l = None
        for l in range(self.layers - 2):
            input = self.activation(self.weights[l] @ input + self.biases[l])

        output = self.activation_output(self.weights[l + 1] @ input + self.biases[l + 1])

        return output

    def forward_pass(self, input):
        self.activated_outputs[0] = input

        l = None
        for l in range(self.layers - 2):
            input = self.weights[l] @ input + self.biases[l]
            input = self.activation(input)
            self.activated_outputs[l + 1] = input

        output = self.weights[l + 1] @ input + self.biases[l + 1]
        output = self.activation_output(output)
        self.activated_outputs[l + 2] = output

    def back_propagation(self, b):   # improve
        cost_derivative, cost = self.loss_function(self, b)

        delta_b = cost_derivative * self.activated_output_derivative(self.activated_outputs[self.layers - 1])
        delta_w = delta_b @ self.activated_outputs[self.layers - 1 - 1].transpose()
        cost_derivative = self.weights[self.layers - 1 - 1].transpose() @ cost_derivative

        self.delta_biases[self.layers - 1 - 1], self.delta_weights[self.layers - 1 - 1] = delta_b, delta_w
        for l in range(self.layers - 1 - 1, 0, -1):
            delta_b = cost_derivative * self.activated_derivative(self.activated_outputs[l])
            delta_w = delta_b @ self.activated_outputs[l - 1].transpose()
            cost_derivative = self.weights[l - 1].transpose() @ cost_derivative

            self.delta_biases[l - 1], self.delta_weights[l - 1] = delta_b, delta_w

        learn = self.optimizer()
        self.weights, self.biases = self.weights - learn * self.delta_weights, self.biases - learn * self.delta_biases

        return cost


    def train(self, training_set=None, epochs=None, batch_size=None, loss_function=None, optimizer=None,
              vectorize=True):
        if vectorize is True and training_set is not None:
            training_set = np.array([[np.array(t[0]).reshape((len(t[0]), 1)), np.array(t[1]).reshape((len(t[1]), 1))]
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

            costs = 0
            t = tm.time()
            for b in batch_set:
                cost = self.back_propagation(b)
                costs += cost
            cost = costs / batch_size
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
    def normal(start=-1, stop=1):
        def initializer(self, start_=start, stop_=stop):
            weights = [np.random.uniform(start_, stop_, (self.shape[i], self.shape[i - 1]))
                       for i in range(1, self.layers)]
            biases = [np.random.uniform(start_, stop_, (self.shape[i], 1)) for i in range(1, self.layers)]

            return np.array(weights, dtype=np.object), np.array(biases, dtype=np.object)

        return initializer

    @staticmethod
    def xavier(start=-1, stop=1, he=1):
        def initializer(self, start_=start, stop_=stop):
            weights = [np.random.uniform(start_ * (he / self.shape[i - 1])**0.5, stop_ * (he / self.shape[i - 1])**0.5,
                                         (self.shape[i], self.shape[i - 1]))
                       for i in range(1, self.layers)]
            biases = [np.random.uniform(start_ * (he / self.shape[i - 1])**0.5, stop_ * (he / self.shape[i - 1])**0.5,
                                        (self.shape[i], 1))
                      for i in range(1, self.layers)]

            return np.array(weights, dtype=np.object), np.array(biases, dtype=np.object) / he

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

            return 1 / (1 + np.e**(-alpha * (x + beta)))

        def activated_derivative(sigmoid_x):

            return alpha * (sigmoid_x * (1 - sigmoid_x))

        return activation, activated_derivative

    @staticmethod
    def relu():
        def activation(x):

            return np.where(x < 0, 0, x)

        def activated_derivative(sigmoid_x):

            return np.where(sigmoid_x < 0, 0, 1)

        return activation, activated_derivative



class Optimizer:
    @staticmethod
    def learning_rate(lr):
        def optimizer():

            return lr

        return optimizer



class LoadNeuralNetwork:
    pass



class SaveNeuralNetwork:
    pass



class PlotGraph:
    def plot_cost_graph(self, nn):
        pass
