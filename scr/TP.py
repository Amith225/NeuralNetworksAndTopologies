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
