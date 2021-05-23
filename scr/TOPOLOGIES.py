from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING: from NEURAL_NETWORKS import CreateArtificialNeuralNetwork

np.NONE = [np.array([None])]


class Initializer:
    # for custom initializer
    def __new__(cls, initializer, __init__=lambda: None, *args, **kwargs):
        cls.initialize = initializer
        cls.args = args
        cls.kwargs = kwargs
        __init__()

        return super(Initializer, cls).__new__(cls, *args, **kwargs)

    def initialize(self, shape, layers):
        pass

    @staticmethod
    def uniform(start=-1, stop=1):
        def initializer(self, shape, layers):
            biases = [np.random.uniform(start, stop, (shape[i], 1)).astype(dtype=np.float32)
                      for i in range(1, layers)]
            weights = [np.random.uniform(start, stop, (shape[i], shape[i - 1])).astype(dtype=np.float32)
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return Initializer(initializer)

    @staticmethod
    def normal(scale=1):
        def initializer(self, shape, layers):
            biases = [(np.random.default_rng().standard_normal((shape[i], 1), dtype=np.float32)) * scale
                      for i in range(1, layers)]
            weights = [(np.random.default_rng().standard_normal((shape[i], shape[i - 1]), dtype=np.float32)) * scale
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return Initializer(initializer)

    @staticmethod
    def xavier(he=1):
        def initializer(self, shape, layers):
            biases = [np.random.default_rng().standard_normal((shape[i], 1),
                                                              dtype=np.float32) * (he / shape[i - 1]) ** 0.5
                      for i in range(1, layers)]
            weights = [np.random.default_rng().standard_normal((shape[i], shape[i - 1]),
                                                               dtype=np.float32) * (he / shape[i - 1]) ** 0.5
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return Initializer(initializer)

    @staticmethod
    def normalized_xavier(he=6):
        def initializer(self, shape, layers):
            biases = [np.random.default_rng().standard_normal((shape[i], 1), dtype=np.float32) *
                      (he / (shape[i - 1] + shape[i])) ** 0.5
                      for i in range(1, layers)]
            weights = [np.random.default_rng().standard_normal((shape[i], shape[i - 1]), dtype=np.float32) *
                       (he / (shape[i - 1] + shape[i])) ** 0.5
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return Initializer(initializer)


class ActivationFunction:
    # for custom activation_function
    def __init__(self, activation, activated_derivative, *args, **kwargs):
        self.activations = activation, activated_derivative
        self.args = args
        self.kwargs = kwargs

    activations = None

    @staticmethod
    def sigmoid(smooth=1, offset=0):
        E = np.float32(np.e)
        SMOOTH = np.float32(smooth)
        OFFSET = np.float32(offset)

        def activation(x):
            return 1 / (1 + E ** (-SMOOTH * (x + OFFSET)))

        def activated_derivative(activated_x):
            return SMOOTH * (activated_x * (1 - activated_x))

        return ActivationFunction(activation, activated_derivative)

    @staticmethod
    def tanh(alpha=1):
        ALPHA = np.float32(alpha)

        def activation(x):
            return np.arctan(ALPHA * x)

        def activated_derivative(activated_x):
            return ALPHA * np.square(np.cos(activated_x))

        return ActivationFunction(activation, activated_derivative)

    @staticmethod
    def relu():
        ONE = np.float32(1)

        def activation(x):
            return x * (x > 0)

        def activated_derivative(activated_x):
            return ONE * (activated_x != 0)

        return ActivationFunction(activation, activated_derivative)

    @staticmethod
    def prelu(leak=0.1):
        ONE = np.float32(1)
        LEAK = np.float32(leak)

        def activation(x):
            return np.where(x > 0, x, LEAK * x)

        def activated_derivative(activated_x):
            return np.where(activated_x == 0, LEAK, ONE)

        return ActivationFunction(activation, activated_derivative)

    @staticmethod
    def elu(alpha=1):
        ONE = np.float32(1)
        E = np.e
        ALPHA = np.float32(alpha)

        def activation(x):
            return np.where(x > 0, x, ALPHA * (E ** x - 1))

        def activated_derivative(activated_x):
            return np.where(activated_x != 0, ONE, activated_x + ALPHA)

        return ActivationFunction(activation, activated_derivative)

    @staticmethod
    def softmax():
        E = np.float32(np.e)

        def activation(x):
            numerator = E ** (x - x.max(axis=1)[:, None])

            return numerator / np.einsum('lij->lj', numerator)[:, None]

        def activated_derivative(activated_x):
            jacobian = -np.einsum('lij,lkj->lik', activated_x, activated_x)
            diag_i = np.diag_indices(jacobian.shape[1])
            jacobian[:, diag_i[1], diag_i[0]] = np.einsum('lij,lij->li', activated_x, 1 - activated_x)

            return jacobian

        return ActivationFunction(activation, activated_derivative)


class LossFunction:
    # for custom loss_function
    def __init__(self, loss_function, *args, **kwargs):
        self.loss_function = loss_function
        self.args = args
        self.kwargs = kwargs

    def loss_function(self, error):
        pass

    @staticmethod
    def mean_square():
        def loss_function(error):
            return np.einsum('lij,lij->', error, error), error

        return LossFunction(loss_function)


class Optimizer:
    # for custom optimizer
    def __init__(self, optimizer, *args, **kwargs):
        self.optimize = optimizer
        self.args = args
        self.kwargs = kwargs

    def optimize(self, layer):
        pass

    @staticmethod
    def gradient_decent(this: 'CreateArtificialNeuralNetwork', learning_rate=0.01):
        LEARNING_RATE = np.float32(learning_rate)

        def optimizer(layer):
            this.delta_biases[layer] *= LEARNING_RATE
            this.delta_weights[layer] *= LEARNING_RATE

        return Optimizer(optimizer)

    @staticmethod
    def momentum(this: 'CreateArtificialNeuralNetwork', learning_rate=0.001, alpha=None):
        if alpha is None: alpha = learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        this.pdb, this.pdw = this.delta_initializer(1)  # pdb -> prev_delta_biases, pdw -> prev_delta_weights

        def optimizer(layer):
            this.delta_biases[layer] = this.pdb[layer] = ALPHA * this.pdb[layer] + \
                                                         LEARNING_RATE * this.delta_biases[layer]
            this.delta_weights[layer] = this.pdw[layer] = ALPHA * this.pdw[layer] + \
                                                          LEARNING_RATE * this.delta_weights[layer]

        return Optimizer(optimizer)

    @staticmethod
    def decay(this: 'CreateArtificialNeuralNetwork', learning_rate=0.01, alpha=None):
        if alpha is None: alpha = 1 / learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        this.decay_count = 0

        def optimizer(layer):
            k = LEARNING_RATE / (1 + this.decay_count / ALPHA)
            this.delta_biases[layer] *= k
            this.delta_weights[layer] *= k

            this.decay_count += 1 / this.batches

        return Optimizer(optimizer)

    @staticmethod
    def nesterov(this: 'CreateArtificialNeuralNetwork', learning_rate=0.001, alpha=None):
        if alpha is None: alpha = learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        this.pdb, this.pdw = this.delta_initializer(1)  # pdb -> prev_delta_biases, pdw -> prev_delta_weights

        def optimizer(layer):
            this.theta[layer] = this.weights[layer] - ALPHA * this.pdw[layer]
            this.delta_biases[layer] = this.pdb[layer] = ALPHA * this.pdb[layer] + \
                                                         LEARNING_RATE * this.delta_biases[layer]
            this.delta_weights[layer] = this.pdw[layer] = ALPHA * this.pdw[layer] + \
                                                          LEARNING_RATE * this.delta_weights[layer]

        return Optimizer(optimizer)

    @staticmethod
    def adagrad(this: 'CreateArtificialNeuralNetwork', learning_rate=0.01, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        EPSILON = np.float32(epsilon)
        this.initialize = True
        this.gsq_b, this.gsq_w = this.delta_initializer(1)  # gsq_b -> grad_square_biases, gsq_w -> grad_square_weights

        def optimizer(layer):
            if this.initialize:
                this.gsq_b, this.gsq_w = this.delta_initializer()
                this.initialize = False

            this.gsq_b[layer] += np.einsum('lij,lij->lij', this.delta_biases[layer], this.delta_biases[layer])
            this.gsq_w[layer] += np.einsum('ij,ij->ij', this.delta_weights[layer], this.delta_weights[layer])

            this.delta_biases[layer] *= LEARNING_RATE / np.sqrt(this.gsq_b[layer] + EPSILON)
            this.delta_weights[layer] *= LEARNING_RATE / np.sqrt(this.gsq_w[layer] + EPSILON)

        return Optimizer(optimizer)

    @staticmethod
    def rmsprop(this: 'CreateArtificialNeuralNetwork', learning_rate=0.001, beta=0.95, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        EPSILON = np.float32(epsilon)
        BETA = np.float32(beta)
        BETA_BAR = np.float32(1 - beta)
        this.initialize = True
        this.gsq_b, this.gsq_w = this.delta_initializer(1)  # gsq_b -> grad_square_biases, gsq_w -> grad_square_weights

        def optimizer(layer):
            if this.initialize:
                this.gsq_b, this.gsq_w = this.delta_initializer()
                this.initialize = False

            this.gsq_b[layer] = BETA * this.gsq_b[layer] + \
                                BETA_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer], this.delta_biases[layer])
            this.gsq_w[layer] = BETA * this.gsq_w[layer] + \
                                BETA_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer], this.delta_weights[layer])

            this.delta_biases[layer] *= LEARNING_RATE / np.sqrt(this.gsq_b[layer] + EPSILON)
            this.delta_weights[layer] *= LEARNING_RATE / np.sqrt(this.gsq_w[layer] + EPSILON)

        return Optimizer(optimizer)

    @staticmethod
    def adadelta(this: 'CreateArtificialNeuralNetwork', learning_rate=0.1, alpha=0.95, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        ALPHA_BAR = np.float32(1 - alpha)
        EPSILON = np.float32(epsilon)
        this.initialize = True
        this.gsq_b, this.gsq_w = this.delta_initializer(1)  # gsq_b -> grad_square_biases, gsq_w -> grad_square_weights
        # dsq_b -> delta_square_biases, dsq_w -> delta_square_weights
        this.dsq_b, this.dsq_w = this.delta_initializer(1)

        def optimizer(layer):
            if this.initialize:
                this.gsq_b, this.gsq_w = this.delta_initializer()
                this.dsq_b, this.dsq_w = this.delta_initializer()
                this.initialize = False

            this.gsq_b[layer] = ALPHA * this.gsq_b[layer] + \
                                ALPHA_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                      this.delta_biases[layer])
            this.gsq_w[layer] = ALPHA * this.gsq_w[layer] + \
                                ALPHA_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer], this.delta_weights[layer])

            this.delta_biases[layer] *= LEARNING_RATE * \
                                        np.sqrt((this.dsq_b[layer] + EPSILON) / (this.gsq_b[layer] + EPSILON))
            this.delta_weights[layer] *= LEARNING_RATE * \
                                         np.sqrt((this.dsq_w[layer] + EPSILON) / (this.gsq_w[layer] + EPSILON))

            this.dsq_b[layer] = ALPHA * this.dsq_b[layer] + \
                                ALPHA_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                      this.delta_biases[layer])
            this.dsq_w[layer] = ALPHA * this.dsq_w[layer] + \
                                ALPHA_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer], this.delta_weights[layer])

        return Optimizer(optimizer)

    @staticmethod
    def adam(this: 'CreateArtificialNeuralNetwork', learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        BETA1 = np.float32(beta1)
        BETA1_BAR = np.float32(1 - beta1)
        BETA2 = np.float32(beta2)
        BETA2_BAR = np.float32(1 - beta2)
        EPSILON = np.float32(epsilon)
        this.decay_count = 0
        this.initialize = True
        this.gsq_b, this.gsq_w = this.delta_initializer(1)  # gsq_b -> grad_square_biases, gsq_w -> grad_square_weights
        this.gb, this.gw = this.delta_initializer(1)  # gsq_b -> grad_biases, gsq_w -> grad_weights

        def optimizer(layer):
            if this.initialize:
                this.gsq_b, this.gsq_w = this.delta_initializer()
                this.gb, this.gw = this.delta_initializer()
                this.initialize = False

            this.gb[layer] = BETA1 * this.gb[layer] + BETA1_BAR * this.delta_biases[layer]
            this.gw[layer] = BETA1 * this.gw[layer] + BETA1_BAR * this.delta_weights[layer]
            this.gsq_b[layer] = BETA2 * this.gsq_b[layer] + \
                                BETA2_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                      this.delta_biases[layer])
            this.gsq_w[layer] = BETA2 * this.gsq_w[layer] + \
                                BETA2_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer], this.delta_weights[layer])

            div_1 = (1 - BETA1 ** (this.epoch + 1))
            div_2 = (1 - BETA2 ** (this.epoch + 1))
            gb_sq = this.gsq_b[layer] / div_2
            gw_sq = this.gsq_w[layer] / div_2

            this.delta_biases[layer] = LEARNING_RATE * this.gb[layer] / div_1 / np.sqrt(gb_sq + EPSILON)
            this.delta_weights[layer] = LEARNING_RATE * this.gw[layer] / div_1 / np.sqrt(gw_sq + EPSILON)

            this.decay_count += 1 / this.batches

        return Optimizer(optimizer)


# database class for training / testing NN
class DataBase:
    def __init__(self, input_set, output_set):
        # class params declaration
        self.input_set = np.array(input_set, dtype=np.float32)
        self.output_set = np.array(output_set, dtype=np.float32)

        # prevent conflicting sizes of input_set and output_set
        if (size := len(self.input_set)) != len(self.output_set):
            raise Exception("Both input_set and output_set should be of same size")

        # class vars initialization
        self.size = size
        self.pointer = 0
        self.block = False
        self.batch_size = None

        # shuffle database
        self.randomize()

    # scale data values within -1 to +1
    def normalize(self):
        input_scale = np.max(np.absolute(self.input_set))
        output_scale = np.max(np.absolute(self.output_set))
        self.input_set /= input_scale
        self.output_set /= output_scale

        return input_scale, output_scale

    # randomly shuffle order of data_sets
    def randomize(self):
        indices = [i for i in range(self.size)]
        np.random.shuffle(indices)
        self.input_set = self.input_set[indices]
        self.output_set = self.output_set[indices]

    # create generator object which yields a set of sequential data with fixed predetermined size everytime its called
    def batch_generator(self, batch_size):
        if self.block:
            raise PermissionError(
                "Access Denied: DataBase currently in use, 'end' previous generator before creating a new one")
        self.block = True
        self.batch_size = batch_size

        def generator():
            while 1:
                i = self.pointer + batch_size
                if i >= self.size:
                    i = self.size
                    r_val = self.__batch(i)
                    self.__return()

                    yield r_val
                    return
                signal = yield self.__batch(i)
                if signal == 'end': return self.__return()
                self.pointer += batch_size

        return generator()

    # returns fixed size of dataset from pointer sequentially
    def __batch(self, i):
        r_val = [self.input_set[self.pointer:i], self.output_set[self.pointer:i]]
        if (filled := i - self.pointer) != self.batch_size:
            vacant = self.batch_size - filled
            r_val[0] = np.append(r_val[0], self.input_set[:vacant]).reshape(
                [self.batch_size, *self.input_set.shape[1:]])
            r_val[1] = np.append(r_val[1], self.output_set[:vacant]).reshape(
                [self.batch_size, *self.output_set.shape[1:]])

        return r_val

    # reinitialize class vars after end of generator
    def __return(self):
        self.pointer = 0
        self.randomize()
        self.block = False
        self.batch_size = None
