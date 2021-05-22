import numpy as np

np.NONE = [np.array([None])]


class Initializer:
    # for custom initializer
    def __init__(self, initializer, *args, **kwargs):
        self.initialize = initializer
        self.args = args
        self.kwargs = kwargs

    def initialize(self, shape, layers):
        pass

    @staticmethod
    def uniform(start=-1, stop=1):
        def initializer(shape, layers):
            biases = [np.random.uniform(start, stop, (shape[i], 1)).astype(dtype=np.float32)
                      for i in range(1, layers)]
            weights = [np.random.uniform(start, stop, (shape[i], shape[i - 1])).astype(dtype=np.float32)
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return Initializer(initializer)

    @staticmethod
    def normal(scale=1):
        def initializer(shape, layers):
            biases = [(np.random.default_rng().standard_normal((shape[i], 1), dtype=np.float32)) * scale
                      for i in range(1, layers)]
            weights = [(np.random.default_rng().standard_normal((shape[i], shape[i - 1]), dtype=np.float32)) * scale
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return Initializer(initializer)

    @staticmethod
    def xavier(he=1):
        def initializer(shape, layers):
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
        def initializer(shape, layers):
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
    def gradient_decent(this, learning_rate=0.01):
        LEARNING_RATE = np.float32(learning_rate)

        def optimizer(layer):
            this.delta_biases[layer] *= LEARNING_RATE
            this.delta_weights[layer] *= LEARNING_RATE

        return Optimizer(optimizer)

    @staticmethod
    def moment(this, learning_rate=0.001, alpha=None):
        if alpha is None:
            alpha = learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        this.prev_delta_biases, this.prev_delta_weights = this.delta_initializer(1)

        def optimizer(layer):
            this.delta_biases[layer] = this.prev_delta_biases[layer] = ALPHA * this.prev_delta_biases[layer] + \
                                                                       LEARNING_RATE * this.delta_biases[layer]
            this.delta_weights[layer] = this.prev_delta_weights[layer] = ALPHA * this.prev_delta_weights[layer] + \
                                                                         LEARNING_RATE * this.delta_weights[layer]

        return Optimizer(optimizer)

    @staticmethod
    def decay(this, learning_rate=0.01, alpha=None):
        if alpha is None:
            alpha = 1 / learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)

        def optimizer(layer):
            k = LEARNING_RATE / (1 + this.decay_count / ALPHA)
            this.delta_biases[layer] *= k
            this.delta_weights[layer] *= k

            this.decay_count += 1 / this.batch_length

        return Optimizer(optimizer)

    @staticmethod
    def nesterov(this, learning_rate=0.001, alpha=None):
        if alpha is None:
            alpha = learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        this.prev_delta_biases, this.prev_delta_weights = this.delta_initializer(1)

        def optimizer(layer):
            this.theta[layer] = this.weights[layer] - ALPHA * this.prev_delta_weights[layer]
            this.delta_biases[layer] = this.prev_delta_biases[layer] = ALPHA * this.prev_delta_biases[layer] + \
                                                                       LEARNING_RATE * this.delta_biases[layer]
            this.delta_weights[layer] = this.prev_delta_weights[layer] = ALPHA * this.prev_delta_weights[layer] + \
                                                                         LEARNING_RATE * this.delta_weights[layer]

        return Optimizer(optimizer)

    @staticmethod
    def adagrad(this, learning_rate=0.01, epsilon=np.e ** -8):
        LEARNING_RATE = np.float32(learning_rate)
        EPSILON = np.float32(epsilon)
        this.initialize = True

        def optimizer(layer):
            if this.initialize:
                this.grad_decay_biases, this.grad_decay_weights = this.delta_initializer()
                this.initialize = False

            this.grad_decay_biases[layer] += np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                       this.delta_biases[layer])
            this.grad_decay_weights[layer] += np.einsum('ij,ij->ij', this.delta_weights[layer],
                                                        this.delta_weights[layer])

            this.delta_biases[layer] *= LEARNING_RATE / np.sqrt(this.grad_decay_biases[layer] + EPSILON)
            this.delta_weights[layer] *= LEARNING_RATE / np.sqrt(this.grad_decay_weights[layer] + EPSILON)

        return Optimizer(optimizer)

    @staticmethod
    def rmsprop(this, learning_rate=0.001, beta=0.009, epsilon=np.e ** -8):
        if beta is None:
            beta = learning_rate
        LEARNING_RATE = np.float32(learning_rate)
        EPSILON = np.float32(epsilon)
        BETA = np.float32(beta)
        BETA_BAR = np.float32(1 - beta)
        this.initialize = True

        def optimizer(layer):
            if this.initialize:
                this.grad_decay_biases, this.grad_decay_weights = this.delta_initializer()
                this.initialize = False

            this.grad_decay_biases[layer] = BETA * this.grad_decay_biases[layer] + \
                                            BETA_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                                 this.delta_biases[layer])
            this.grad_decay_weights[layer] = BETA * this.grad_decay_weights[layer] + \
                                             BETA_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer],
                                                                  this.delta_weights[layer])

            this.delta_biases[layer] *= LEARNING_RATE / np.sqrt(this.grad_decay_biases[layer] + EPSILON)
            this.delta_weights[layer] *= LEARNING_RATE / np.sqrt(this.grad_decay_weights[layer] + EPSILON)

        return Optimizer(optimizer)

    @staticmethod
    def adadelta(this, alpha=0.01, epsilon=np.e ** -8):
        ALPHA = np.float32(alpha)
        ALPHA_BAR = np.float32(1 - alpha)
        EPSILON = np.float32(epsilon)
        this.initialize = True

        def optimizer(layer):
            if this.initialize:
                this.grad_decay_biases, this.grad_decay_weights = this.delta_initializer()
                this.delta_decay_biases, this.delta_decay_weights = this.delta_initializer()
                this.initialize = False

            this.grad_decay_biases[layer] = ALPHA * this.grad_decay_biases[layer] + \
                                            ALPHA_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                                 this.delta_biases[layer])
            this.grad_decay_weights[layer] = ALPHA * this.grad_decay_weights[layer] + \
                                             ALPHA_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer],
                                                                  this.delta_weights[layer])

            this.delta_biases[layer] =\
                this.delta_biases[layer] *\
                np.sqrt(this.delta_decay_biases[layer] + EPSILON) / np.sqrt(this.grad_decay_biases[layer] + EPSILON)
            this.delta_weights[layer] =\
                this.delta_weights[layer] *\
                np.sqrt(this.delta_decay_weights[layer] + EPSILON) / np.sqrt(this.grad_decay_weights[layer] + EPSILON)

            this.delta_decay_biases[layer] = ALPHA * this.delta_decay_biases[layer] + \
                                             ALPHA_BAR * np.einsum('lij,lij->lij', this.delta_biases[layer],
                                                                   this.delta_biases[layer])
            this.delta_decay_weights[layer] = ALPHA * this.delta_decay_weights[layer] + \
                                              ALPHA_BAR * np.einsum('ij,ij->ij', this.delta_weights[layer],
                                                                    this.delta_weights[layer])

        return Optimizer(optimizer)


class DataBase:
    def __init__(self, input_set, output_set):
        self.input_set = np.array(input_set, dtype=np.float32)
        self.output_set = np.array(output_set, dtype=np.float32)

        if (size := len(self.input_set)) != len(self.output_set):
            raise Exception("Both input_set and output_set should be of same size")

        self.size = size
        self.pointer = 0
        self.block = False
        self.batch_size = None

        self.randomize()

    def normalize(self):
        input_scale = np.max(self.input_set)
        output_scale = np.max(self.output_set)
        self.input_set /= input_scale
        self.output_set /= output_scale

        return input_scale, output_scale

    def randomize(self):
        indices = [i for i in range(self.size)]
        np.random.shuffle(indices)
        self.input_set = self.input_set[indices]
        self.output_set = self.output_set[indices]

    def batch_generator(self, batch_size):
        if self.block:
            raise PermissionError(
                "Access Denied: DataBase currently in use, end previous generator before creating a new one")
        self.block = True
        self.batch_size = batch_size

        def generator():
            while 1:
                i = self.pointer + batch_size
                if i >= self.size:
                    i = self.size
                    r_val = self.batch(i)
                    self.return_()

                    yield r_val
                    return
                signal = yield self.batch(i)
                if signal == 'end':
                    return self.return_()
                self.pointer += batch_size

        return generator()

    def batch(self, i):
        r_val = [self.input_set[self.pointer:i], self.output_set[self.pointer:i]]
        if (filled := i - self.pointer) != self.batch_size:
            vacant = self.batch_size - filled
            r_val[0] = np.append(r_val[0], self.input_set[:vacant]).reshape(
                [self.batch_size, *self.input_set.shape[1:]])
            r_val[1] = np.append(r_val[1], self.output_set[:vacant]).reshape(
                [self.batch_size, *self.output_set.shape[1:]])
        return r_val

    def return_(self):
        self.pointer = 0
        self.randomize()
        self.block = False
        self.batch_size = None
