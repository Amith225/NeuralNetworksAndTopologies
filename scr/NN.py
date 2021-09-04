import cProfile as cp
import time as tm

import numpy as np

import print_vars as pv


class CreateANN:
    DELAY_INTERVAL = 0.25

    def __init__(self, shape,
                 initializer,
                 activation_function,
                 output_activation_function):
        self.shape = tuple(shape)
        self.initializer = initializer
        self.activation, self.activated_derivative = activation_function()
        self.output_activation, self.output_activated_derivative = output_activation_function()

        self.num_layers = len(self.shape)
        self.biases, self.weights = self.initializer(self.shape)
        self.theta = self.weights.copy()
        self.biases_ones = np.NONE + [np.ones_like(bias, dtype=np.float32) for bias in self.biases[1:]]

        self.train_database = None
        self.epochs = 1
        self.batch_size = 32
        self.loss_function = None
        self.optimizer = None
        self.num_batches = None
        self.costs = [[]]
        self.outputs, self.target = list(range(self.num_layers)), None
        self.loss, self.loss_derivative = None, self.outputs.copy()
        self.delta_biases, self.delta_weights = None, None

    def __forward_pass(self, layer=1):
        if layer < self.num_layers - 1:
            self.__fire(layer, self.activation)
            self.__forward_pass(layer + 1)
        else:
            self.__fire(layer, self.output_activation)

    def process(self, inputs):
        self.outputs[0] = inputs
        self.__forward_pass()

        return self.outputs[-1]

    def __fire(self, layer, activation):
        self.outputs[layer] = activation(self.weights[layer] @ self.outputs[layer - 1] + self.biases[layer])

    def __wire(self, layer):
        self.biases[layer] -= (self.delta_biases[layer] * self.biases_ones[layer])[0]
        self.weights[layer] -= self.delta_weights[layer]
        self.theta = self.weights.copy()

    def __back_propagation(self, activated_derivative, layer=-1):
        if layer <= -self.num_layers: return
        self.delta_biases[layer] = self.loss_derivative[layer] * activated_derivative(self.outputs[layer])
        np.einsum('lkj,lij->ik', self.outputs[layer - 1], self.delta_biases[layer], out=self.delta_weights[layer])
        self.loss_derivative[layer - 1] = self.theta[layer].transpose() @ self.loss_derivative[layer]
        self.optimizer(layer)
        self.__wire(layer)
        self.__back_propagation(self.activated_derivative, layer - 1)

    def delta_initializer(self):
        delta_biases = np.NONE + [(np.zeros((self.batch_size, self.shape[i], 1), dtype=np.float32))
                                  for i in range(1, self.num_layers)]
        delta_weights = self.initializer(self.shape)[1]

        return delta_biases, delta_weights

    def train(self, epochs=None,
              batch_size=None,
              train_database=None,
              loss_function=None,
              optimizer=None,
              profile=False):
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if train_database is not None: self.train_database = train_database
        if loss_function is not None: self.loss_function = loss_function
        if optimizer is not None: self.optimizer = optimizer

        self.delta_biases, self.delta_weights = self.delta_initializer()
        self.loss_derivative = [(np.zeros((self.batch_size, self.shape[i], 1), dtype=np.float32))
                                for i in range(0, self.num_layers)]

        if not profile:
            costs = [0]
            tot_time = 0
            delay = self.DELAY_INTERVAL
            self.num_batches = int(np.ceil(self.train_database.size / self.batch_size))
            last_epoch = self.epochs - 1
            for epoch in range(self.epochs):
                cost = 0
                time = tm.time()
                batch_generator = self.train_database.batch_generator(self.batch_size)
                for batch in range(self.num_batches):
                    self.outputs[0], self.target = batch_generator.__next__()
                    self.__forward_pass()
                    self.loss, self.loss_derivative[-1] = self.loss_function(self.outputs[-1], self.target)
                    self.__back_propagation(self.output_activated_derivative)
                    cost += self.loss
                time = tm.time() - time
                tot_time += time
                cost /= self.train_database.size
                costs.append(cost)
                if tot_time > delay or epoch == last_epoch:
                    delay += self.DELAY_INTERVAL
                    print(end='\r')
                    print(pv.CBOLD + pv.CBLUE + pv.CURL + f'epoch:{epoch}' + pv.CEND,
                          pv.CYELLOW + f'cost:{cost}', f'time:{time}' + pv.CEND,
                          pv.CBOLD + pv.CITALIC + pv.CBEIGE + f'cost_reduction:{(costs[-2] - cost)}' + pv.CEND,
                          pv.CBOLD + f'eta:{tot_time / (epoch + 1) * (self.epochs - epoch - 1)}',
                          pv.CEND, end='')
            print('\n' + pv.CBOLD + pv.CRED2 + f'tot_time:{tot_time}', f'avg_time:{tot_time / self.epochs}' + pv.CEND)
            self.costs.append(costs[1:])
        else:
            cp.runctx("self.train()", globals=globals(), locals=locals())


class SaveNN:
    pass


class LoadNN:
    pass
