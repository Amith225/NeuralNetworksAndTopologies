import numpy as np

np.NONE = [np.array([None])]


class Initializer:
    @staticmethod
    def uniform(start=-1, stop=1):
        def initializer(shape):
            layers = len(shape)
            biases = [np.random.uniform(start, stop, (shape[i], 1)).astype(dtype=np.float32)
                      for i in range(1, layers)]
            weights = [np.random.uniform(start, stop, (shape[i], shape[i - 1])).astype(dtype=np.float32)
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return initializer

    @staticmethod
    def normal(scale=1):
        def initializer(shape):
            layers = len(shape)
            sn = np.random.default_rng().standard_normal
            biases = [(sn((shape[i], 1), dtype=np.float32)) * scale for i in range(1, layers)]
            weights = [(sn((shape[i], shape[i - 1]), dtype=np.float32)) * scale for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return initializer

    @staticmethod
    def xavier(he=1):
        def initializer(shape):
            layers = len(shape)
            sn = np.random.default_rng().standard_normal
            biases = [sn((shape[i], 1), dtype=np.float32) * (he / shape[i - 1]) ** 0.5 for i in range(1, layers)]
            weights = [sn((shape[i], shape[i - 1]), dtype=np.float32) * (he / shape[i - 1]) ** 0.5
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return initializer

    @staticmethod
    def normalized_xavier(he=6):
        def initializer(shape):
            layers = len(shape)
            sn = np.random.default_rng().standard_normal
            biases = [sn((shape[i], 1), dtype=np.float32) * (he / (shape[i - 1] + shape[i])) ** 0.5
                      for i in range(1, layers)]
            weights = [sn((shape[i], shape[i - 1]), dtype=np.float32) * (he / (shape[i - 1] + shape[i])) ** 0.5
                       for i in range(1, layers)]

            return np.NONE + biases, np.NONE + weights

        return initializer


class ActivationFunction:
    @staticmethod
    def sigmoid(smooth=1, offset=0):
        ONE = np.float32(1)
        E = np.float32(np.e)
        SMOOTH = np.float32(smooth)
        OFFSET = np.float32(offset)

        def activation(x):
            return ONE / (ONE + E ** (-SMOOTH * (x - OFFSET)))

        def activated_derivative(activated_x):
            return SMOOTH * (activated_x * (ONE - activated_x))

        return lambda: (activation, activated_derivative)

    @staticmethod
    def tanh(alpha=1):
        ALPHA = np.float32(alpha)

        def activation(x):
            return np.arctan(ALPHA * x)

        def activated_derivative(activated_x):
            return ALPHA * np.square(np.cos(activated_x))

        return lambda: (activation, activated_derivative)

    @staticmethod
    def relu():
        ONE = np.float32(1)

        def activation(x):
            return x * (x > 0)

        def activated_derivative(activated_x):
            return ONE * (activated_x != 0)

        return lambda: (activation, activated_derivative)

    @staticmethod
    def prelu(leak=0.01):
        if leak < 0: raise ValueError("parameter 'leak' cannot be less than zero")
        ONE = np.float32(1)
        LEAK = np.float32(leak)

        def activation(x):
            return np.where(x > 0, x, LEAK * x)

        def activated_derivative(activated_x):
            return np.where(activated_x <= 0, LEAK, ONE)

        return lambda: (activation, activated_derivative)

    @staticmethod
    def elu(alpha=1):
        if alpha < 0: raise ValueError("parameter 'alpha' cannot be less than zero")
        ONE = np.float32(1)
        E = np.e
        ALPHA = np.float32(alpha)

        def activation(x):
            return np.where(x > 0, x, ALPHA * (E ** x - 1))

        def activated_derivative(activated_x):
            return np.where(activated_x <= 0, activated_x + ALPHA, ONE)

        return lambda: (activation, activated_derivative)

    @staticmethod
    def softmax():
        E = np.float32(np.e)

        def activation(x):
            numerator = E ** (x - x.max(axis=1, keepdims=1))

            return numerator / numerator.sum(axis=1, keepdims=1)

        def activated_derivative(activated_x):
            jacobian = activated_x @ activated_x.transpose(0, 2, 1)
            diag_i = np.diag_indices(jacobian.shape[1])
            jacobian[:, [diag_i[1]], [diag_i[0]]] = (activated_x * (1 - activated_x)).transpose(0, 2, 1)

            return jacobian.sum(axis=2, keepdims=1)

        return lambda: (activation, activated_derivative)

    @staticmethod
    def softplus():
        E = np.float32(np.e)
        ONE = np.float32(1)

        def activation(x):
            return np.log(ONE + E ** x)

        def activated_derivative(activated_x):
            return ONE - E ** -activated_x

        return lambda: (activation, activated_derivative)


class LossFunction:
    @staticmethod
    def mean_square():
        def loss_function(output, target):
            loss = output - target

            return np.einsum('lij,lij->', loss, loss), loss

        return loss_function


class Optimizer:
    @staticmethod
    def gradient_decent(this, learning_rate=0.01):
        LEARNING_RATE = np.float32(learning_rate)

        def optimizer(layer):
            this.delta_biases[layer] *= LEARNING_RATE
            this.delta_weights[layer] *= LEARNING_RATE

        return optimizer

    @staticmethod
    def momentum(this, learning_rate=0.01, alpha=None):
        if alpha is None: alpha = learning_rate / 10
        LEARNING_RATE = np.float32(learning_rate)
        ALPHA = np.float32(alpha)
        this.pdb = list(range(this.num_layers))  # pdb -> prev_delta_biases
        this.pdw = this.pdb.copy()  # pdw -> prev_delta_weights

        def optimizer(layer):
            this.delta_biases[layer] = this.pdb[layer] = ALPHA * this.pdb[layer] + \
                                                         LEARNING_RATE * this.delta_biases[layer]
            this.delta_weights[layer] = this.pdw[layer] = ALPHA * this.pdw[layer] + \
                                                          LEARNING_RATE * this.delta_weights[layer]

        return optimizer


class DataBase:
    def __init__(self, input_set, target_set):
        if (size := len(input_set)) != len(target_set):
            raise Exception("Both input and output set should be of same size")

        self.input_set = np.array(input_set, dtype=np.float32)
        self.target_set = np.array(target_set, dtype=np.float32)

        self.size = size
        self.pointer = 0
        self.block = False
        self.batch_size = None
        self.indices = list(range(self.size))

    def normalize(self):
        input_scale = np.max(np.absolute(self.input_set))
        target_scale = np.max(np.absolute(self.target_set))
        self.input_set /= input_scale
        self.target_set /= target_scale

    def randomize(self):
        np.random.shuffle(self.indices)
        self.input_set = self.input_set[self.indices]
        self.target_set = self.target_set[self.indices]

    def batch_generator(self, batch_size):
        if self.block:
            raise PermissionError(
                "Access Denied: DataBase currently in use, 'end' previous generator before creating a new one")
        self.block = True
        self.batch_size = batch_size
        self.randomize()

        def generator():
            while 1:
                if (i := self.pointer + batch_size) >= self.size:
                    i = self.size
                    r_val = self.__batch(i)
                    self.__return()
                    yield r_val
                    return
                signal = yield self.__batch(i)
                if signal == -1: return self.__return()
                self.pointer += batch_size

        return generator()

    def __batch(self, i):
        r_val = [self.input_set[self.pointer:i], self.target_set[self.pointer:i]]
        if (filled := i - self.pointer) != self.batch_size:
            vacant = self.batch_size - filled
            r_val[0] = \
                np.append(r_val[0], self.input_set[:vacant]).reshape([self.batch_size, *self.input_set.shape[1:]])
            r_val[1] = \
                np.append(r_val[1], self.target_set[:vacant]).reshape([self.batch_size, *self.target_set.shape[1:]])

        return r_val

    def __return(self):
        self.pointer = 0
        self.block = False
        self.batch_size = None
