import numpy as np
import numpy as _np
import os as _os
import numexpr as _ne
import typing as _tp

from typing import *

from NeuralNetworks import *
from Topologies import *


class CreateCNN(ArtificialNeuralNetwork):
    class CORRELATION_TYPE:
        VALID = 0; FULL = 1; SAME = 2

    class POOLING_TYPE:
        MAX = 0; MEAN = 1

    def __init__(self, conv_shape, kr_initializer, kr_activations, kr_strides,
                 correlation_type, padding, pooling,
                 ann_shape, wb_initializer, wb_activations):
        self.convShape = self.format_conv_shape(conv_shape)
        self.krInitializer = kr_initializer
        self.krActivations = []
        self.krActivationDerivatives = []
        for activation_function in tuple(kr_activations):
            self.krActivations.append(activation_function.activation)
            self.krActivationDerivatives.append(activation_function.activatedDerivative)
        self.krStrides = self.format_strides(kr_strides)
        self.correlation_type = correlation_type
        self.padding = self.format_padding(padding)
        self.pooling = self.format_pooling(pooling)

        self.kernels_shape = self.format_kernels_shape()
        self.conv_output_shapes = [self.convShape[0][0]]
        self.pooled_output_shapes = [self.convShape[0][0]]
        [(self.conv_output_shapes.append(self.format_output_shape(s)),
          self.pooled_output_shapes.append(self.format_pooled_shape()))
         for s in self.kernels_shape[1:]]
        # self.krBiases_list, self.kernels_list = self.krInitializer(self.convShape, self.conv_output_shapes)

        wb_shape = ann_shape
        super(CreateCNN, self).__init__(wb_shape, wb_initializer, wb_activations)

    def format_kernels_shape(self):
        formatted_kernels_shape = []
        prev_layer_len = 1
        for i, layer in enumerate(self.convShape):
            new_shape = []
            layer_len = 0
            for shape in layer:
                if i != 0:
                    new_shape.append((shape[0], shape[1], prev_layer_len, shape[2]))
                else:
                    new_shape.append((*shape, prev_layer_len))
                layer_len += shape[2]
            prev_layer_len = layer_len
            formatted_kernels_shape.append(new_shape)

        return formatted_kernels_shape

    def format_output_shape(self, kernel_shapes):
        conv_output_shapes = [0, 0, 0]
        for krn_shape in kernel_shapes:
            shape = self.__output_shape_function(self.pooled_output_shapes[-1], krn_shape)
            if (x := shape[0]) > conv_output_shapes[0]: conv_output_shapes[0] = x
            if (y := shape[1]) > conv_output_shapes[1]: conv_output_shapes[1] = y
            conv_output_shapes[2] += shape[2]

        return tuple(conv_output_shapes)

    def format_pooled_shape(self):
        return self.__pooled_shape_function(self.conv_output_shapes[-1], self.pooling[len(self.conv_output_shapes) - 2])

    @staticmethod
    def __pooled_shape_function(out_shape, pool):
        r_val = np.ceil((out_shape[0] - pool[1]) / pool[2]).astype(int) + 1, \
                np.ceil((out_shape[1] - pool[1]) / pool[2]).astype(int) + 1

        return r_val

    def __output_shape_function(self, inp_shape, krn_shape):
        pad = self.padding[0] * 2
        if self.correlation_type == self.CORRELATION_TYPE.VALID:
            r_val = np.ceil((inp_shape[0] + pad - krn_shape[0]) / self.krStrides[0]).astype(int) + 1, \
                    np.ceil((inp_shape[1] + pad - krn_shape[1]) / self.krStrides[1]).astype(int) + 1
        elif self.correlation_type == self.CORRELATION_TYPE.FULL:
            r_val = np.ceil((inp_shape[0] + pad + krn_shape[0]) / self.krStrides[0]).astype(int) - 1, \
                    np.ceil((inp_shape[1] + pad + krn_shape[1]) / self.krStrides[1]).astype(int) - 1
        elif self.correlation_type == self.CORRELATION_TYPE.SAME:
            r_val = inp_shape
        else:
            raise ValueError("Invalid Correlation Type")

        return *r_val, krn_shape[3]

    @staticmethod
    def format_padding(padding):
        return padding

    @staticmethod
    def format_pooling(pooling):
        return pooling

    @staticmethod
    def format_strides(strides):
        return strides

    @staticmethod
    def format_conv_shape(conv_shape):
        formatted_conv_shape = []
        for layer in conv_shape:
            new_shape = []
            try:
                for k in layer:
                    try:
                        if (length := len(k)) == 2:
                            new_shape.append((*k, 1))
                        elif length == 3:
                            new_shape.append(k)
                        else:
                            raise ValueError("length of shapes in 'conv_shape' should be either 1, 2 or 3.\n"
                                             f"but given length {length}")
                    except TypeError:
                        new_shape.append((k, k, 1))
                formatted_conv_shape.append(new_shape)
            except TypeError:
                formatted_conv_shape.append([(layer, layer, 1)])

        return tuple(formatted_conv_shape)


"""
class ProxyCreateCNN(AbstractNeuralNetwork):
    # bug at stride array, apply padding
    def cross_correlate(self, array, layer):
        kernel = self.kernels_list[layer]
        if self.correlation_type == self.CORRELATION_TYPE_VALID:
            strides = array.strides[0] * self.stride, array.strides[1] * self.stride, *array.strides
            crossed = np.lib.stride_tricks.as_strided(array, (*self.kernel_output_shapes[layer], *kernel.shape),
                                                      strides)

            return (crossed * kernel).sum(axis=(2, 3))

    def __forward_pass(self, layer=1):
        self.__fire(layer)
        if layer < self.num_kernels - 1:
            self.__forward_pass(layer + 1)
        else:
            self.wbOutputs[0] = self.krOutputs[-1].reshape((-1, self.wbShape[0], 1))
            # noinspection PyUnresolvedReferences
            self._CreateANN__forward_pass()

    def process(self, inputs):
        self.krOutputs[0] = inputs
        self.__forward_pass()

        return self.wbOutputs[-1]

    def __fire(self, layer):
        out = np.zeros((self.krOutputs[0].shape[0], *self.kernel_output_shapes[layer]), dtype=np.float32)
        for i, o in enumerate(self.krOutputs[layer - 1]):
            out[i] = self.cross_correlate(o, layer)
            # print(np.allclose(sg.correlate2d(o, self.kernels_list[layer], mode='valid'), out[i], rtol=1e-4, atol=1e-4))
        self.krOutputs[layer] = out
"""


class KRInitializer:
    @staticmethod
    def uniform(start=-1, stop=1):
        def initializer(shape, conv_output_shapes):
            print(shape)
            print(conv_output_shapes)
            exit()
            kernel_biases = [np.random.uniform(start, stop, [*s]).astype(dtype=np.float32)
                             for s in conv_output_shapes[1:]]
            kernels = [np.random.uniform(start, stop, [*s]).astype(dtype=np.float32) for s in shape[1:]]

            return np.NONE + kernel_biases, np.NONE + kernels

        return initializer

    @staticmethod
    def normal(scale=1):
        def initializer(shape, kernel_output_shapes):
            sn = np.random.default_rng().standard_normal
            kernel_biases = [(sn([*s], dtype=np.float32)) * scale for s in kernel_output_shapes[1:]]
            kernels = [(sn([*s], dtype=np.float32)) * scale for s in shape[1:]]

            return np.NONE + kernel_biases, np.NONE + kernels

        return initializer


# noinspection PyPep8Naming
class KRActivationFunction:  # main class
    # WBActivationFunction class and methods / params for custom activation function
    def __init__(self, activation, activated_derivative):
        self.__activation = activation
        self.__activated_derivative = activated_derivative

    def activation(self, x: np.ndarray) -> np.ndarray:
        return self.__activation(x)

    def activated_derivative(self, activated_x: np.ndarray) -> np.ndarray:
        return self.__activated_derivative(activated_x)

    # Pre-Defined activation functions
    @staticmethod
    def sigmoid(smooth: Union[int, float] = 1, offset: Union[int, float] = 0) -> "KRActivationFunction":
        # constants
        ONE = np.float32(1)
        E = np.float32(np.e)
        SMOOTH = np.float32(smooth)
        OFFSET = np.float32(offset)

        def activation(x: np.ndarray) -> np.ndarray: return ONE / (ONE + E ** (-SMOOTH * (x - OFFSET)))

        def activated_derivative(activated_x: np.ndarray) -> np.ndarray:
            return SMOOTH * (activated_x * (ONE - activated_x))

        return KRActivationFunction(activation, activated_derivative)



class ProxyDataBase:
    def __init__(self, inputSet: _tp.Iterable and _tp.Sized,  # input signal
                 targetSet: _tp.Iterable and _tp.Sized,  # desired output signal
                 normalize: _tp.Union[int, float] = 0):
        if (size := len(inputSet)) != len(targetSet):
            raise Exception("Both input and output set should be of same size")

        inputSet = self.normalize(_np.array(inputSet, dtype=_np.float32), normalize)
        targetSet = self.normalize(_np.array(targetSet, dtype=_np.float32), normalize)
        fTmp = tf.NamedTemporaryFile(delete=False)
        self.cacheInputSet = fTmp.name + ".npy"
        _np.save(self.cacheInputSet, inputSet, allow_pickle=False)
        fTmp = tf.NamedTemporaryFile(delete=False)
        self.cacheTargetSet = fTmp.name + ".npy"
        _np.save(self.cacheTargetSet, targetSet, allow_pickle=False)

        self.size: int = size
        self.inpShape = inputSet.shape[1]
        self.tarShape = targetSet.shape[1]
        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

    # normalize input and target sets within the range of -scale to +scale
    @staticmethod
    def normalize(data, scale: _tp.Union[int, float] = 1) -> "_np.ndarray":
        if scale != 0:  # do not normalize if scale is zero
            scale = _ne.evaluate("abs(data) * scale", local_dict={'data': data, 'scale': scale}).max()

            return data / scale
        else:
            return data

    # save database as NPZ file
    def save(self, fname: str = None) -> str:
        if fname is None:
            fname = 'db'
        fpath = _os.getcwd() + '\\DataSets\\'
        spath = fpath + fname + f's{self.size}i{self.inpShape}o{self.tarShape}'

        i = 0
        nSpath = spath
        while 1:
            if i != 0:
                nSpath = spath + ' (' + str(i) + ')'
            if _os.path.exists(nSpath + '.nndb' + '.npz'):
                i += 1
            else:
                spath = nSpath
                break
        _os.makedirs(fpath, exist_ok=True)
        _np.savez_compressed(spath + '.nndb', self.inputSet, self.targetSet)

        return spath

    # load a database file
    @staticmethod
    def load(file: str, normalize: _tp.Union[int, float] = 0) -> "DataBase":
        if file:
            if not _os.path.dirname(file):
                file = _os.getcwd() + '\\DataSets\\' + file
        else:
            raise FileExistsError("file not given")

        if file[-4:] != '.npz':
            raise ValueError(f"file type must be that of 'NPZ' but given {file}")
        nnLoader = _np.load(file)

        return DataBase(nnLoader['arr_0'], nnLoader['arr_1'], normalize)
