from typing import *

import numpy as np

from .denseNN import DenseNN
from ..tools import *
from ..Topologies import *

from testNumba import foo, foo2


class DenseNNForCNN(DenseNN):
    def __new__(cls, *args, **kwargs):
        cls.train = cls.test = lambda: NotImplemented
        return super(DenseNNForCNN, cls).__new__(cls)

    def forwardPass_(self, inputs) -> "np.ndarray":
        self.outputs[0] = inputs.reshape(-1, *self.shape.INPUT)
        self._forwardPass()
        return self.outputs[-1]

    def backPropagate_(self, deltaLoss):
        self.deltaLoss[-1] = deltaLoss
        self._backPropagate()


class ConvolutionalNN():
    def _forwardPass(self, layer=1):
        self._fire(layer)
        if layer < self.shape.LAYERS - 1:
            self._forwardPass(layer + 1)
        else:
            self.outputs.append(self.ann.forwardPass_(self.outputs[layer]))
            self.deltaLoss.append(None)

    def _backPropagate(self, layer=-1):
        if layer < -self.shape.LAYERS: return
        if layer == -1:
            self.ann.backPropagate_(self.deltaLoss[-1])
            self.deltaLoss[-2] = self.ann.deltaLoss[0].reshape((-1, *self.outputShape[-1]))
        else:
            # optimizer
            print(self.deltaLoss[layer].shape, self.outputs[layer].shape)
            self._wire(layer + 1)
            exit()
        self._backPropagate(layer - 1)

    def _fire(self, layer):
        self.outputs[layer] = self.activations[layer](self.crossCorrelate(layer) + self.biasesList[layer])

    def _wire(self, layer):
        self.biasesList[layer] -= self.deltaBiases[layer]
        for k in range(len(self.kernelsList[layer])):
            self.kernelsList[layer][k] -= self.kernelsList[layer][k]

    def _findOutputShape(self):
        return self.__outputShape

    def _initializeWeightsBiasesDelta(self):
        self.biasesList = self.initializer(self.outputShape)
        kernelsShapeFlat = [(1, *self.shape[0])]
        for i, s in enumerate(self.shape[1:]):
            if not iterable(s[0]):
                kernelsShapeFlat.append((s[0], self.outputShape[i][0], *s[1:]))
            else:
                kernelsShapeFlat.extend([(ss[0], self.outputShape[i][0], *ss[1:]) for ss in s])
        kernelsListFlat = self.initializer(kernelsShapeFlat)
        x = 0
        self.kernelsList = [[kernelsListFlat[i + x]] if not iterable(s[0]) else
                            [kernelsListFlat[i + (x := x + y)] for y in range(len(s))]
                            for i, s in enumerate(self.shape)]
        self.deltaBiases = [np.zeros_like(bias) for bias in self.biasesList]
        self.deltaKernels = [np.zeros_like(kernel) if type(kernel) == np.ndarray else [np.zeros_like(k) for k in kernel]
                             for kernel in self.kernelsList]

    def train(self,
              epochs: "int" = None,
              batchSize: "int" = None,
              trainDataBase: "DataBase" = None,
              optimizer=None, annOptimizer=None,
              profile: "bool" = False,
              test: Union["bool", "DataBase"] = None):
        if annOptimizer is not None: self.ann.optimizer = annOptimizer
        super(ConvolutionalNN, self).train(epochs, batchSize, trainDataBase, optimizer, profile, test)


    class PoolingType:
        MAX = 0
        MEAN = 1

    class CorrelationType:
        VALID = 0
        FULL = 1
        SAME = 2

    def __init__(self,
                 shape: "Shape", annShape: "Shape",
                 initializer: "Abstract", annInitializer: "Abstract",
                 activators: "Activators", annActivators: "Activators",
                 lossFunction: "Abstract", annLossFunction: "Abstract",
                 strides,
                 poolingStride,
                 poolingShape,
                 poolingType: "Types",
                 poolingCorrelationType: "Types",
                 correlationType: "Types"):
        self.strides = self.__formatStrides(strides, shape)
        self.poolingStride = self.__formatStrides(poolingStride, shape)
        self.poolingShape = poolingShape
        self.poolingType = poolingType(shape.LAYERS - 1)
        self.poolingCorrelationType = poolingCorrelationType(shape.LAYERS - 1)
        self.correlationType = correlationType(shape.LAYERS - 1)
        self.__outputShape, self.padding, self.outputShapeBeforePool, self.poolingPadding = \
            self.__findOutputShapePadding(shape)
        super(ConvolutionalNN, self).__init__(shape, initializer, activators, lossFunction)

        self.poolPseudoKernel = [np.NAN] + [np.zeros((self.outputShape[i + 1][0], *s))[None]
                                            for i, s in enumerate(self.poolingShape)]

        annShape = Shape(np.prod(self.outputShape[-1]), *annShape)
        self.ann = DenseNNForCNN(annShape, annInitializer, annActivators, annLossFunction)

    def crossCorrelate(self, layer):
        crossedStack = np.zeros((self.outputs[layer - 1].shape[0], *self.outputShapeBeforePool[layer]),
                                dtype=np.float32)
        a = crossedStack.shape[1] // len(self.kernelsList[layer])
        for j, k in enumerate(self.kernelsList[layer]):
            crossedStack[:, j:j + a] = self.__crossCorrelate(self.outputs[layer - 1], self.padding[layer][j],
                                                             self.outputShapeBeforePool[layer], self.strides[layer], k)
        pooled = self.__crossCorrelate(crossedStack, self.poolingPadding[layer],
                                       self.outputShape[layer], self.poolingStride[layer], self.poolPseudoKernel[layer],
                                       poolingType=self.poolingType[layer])

        return pooled

    @staticmethod
    def __crossCorrelate(array, pad, outShape, stride, kernel, poolingType=None):
        paddedArray = np.pad(array, ((0, 0), (0, 0), *pad))
        shape = paddedArray.shape[0], kernel.shape[1], *outShape[1:], *kernel.shape[2:]
        stride = *paddedArray.strides[:2], *[stride[i] * paddedArray.strides[i] for i in (-2, -1)], \
                 *paddedArray.strides[-2:]
        strided = np.lib.stride_tricks.as_strided(paddedArray, shape, stride)
        if poolingType is None:
            # crossed = np.einsum("mlxyij, klij -> mkxy", strided, kernel, optimize=True)
            crossed = foo(strided, kernel)
        else:
            if poolingType == ConvolutionalNN.PoolingType.MAX:
                # crossed = strided.max(axis=(-2, -1))
                crossed = foo2(strided)
            elif poolingType == ConvolutionalNN.PoolingType.MEAN:
                crossed = strided.mean(axis=(-2, -1))
            else:
                raise ValueError("non valid poolingType")

        return crossed

    # todo: optimize and make more readable, and damn checking is so long
    def __findOutputShapePadding(self, shape):
        outputShape = [shape[0]]
        outShapeBeforePool = [shape[0]]
        padding = [np.NAN]
        poolingPadding = [np.NAN]
        for i, krShape in enumerate(shape[1:]):
            i += 1
            if iterable(krShape[0]):
                subShape = []
                subPad = []
                depth = 0
                for j, subKrShape in enumerate(krShape):
                    outShape, pad = self.__findOutPad(outputShape[-1], self.strides[i], subKrShape,
                                                      self.correlationType[i])
                    subShape.append((subKrShape[0], *outShape))
                    subPad.append(pad)
                    depth += subKrShape[0]
                outShape = depth, *np.maximum.reduce(subShape)[1:]
                subPad = [self.__findPadFromOut(outputShape[-1], outShape[1:], self.strides[i], skr) for skr in krShape]
                padding.append([self.__formatPad(pad) for pad in subPad])
            else:
                outShape, pad = self.__findOutPad(outputShape[-1], self.strides[i], krShape, self.correlationType[i])
                outShape = krShape[0], *outShape
                padding.append([self.__formatPad(pad)])
            depth = outShape[0]
            outShapeBeforePool.append(outShape)
            outShape, poolingPad = self.__findOutPad(outShape, self.poolingStride[i], self.poolingShape[i - 1],
                                                     self.poolingCorrelationType[i])
            outputShape.append((depth, *outShape))
            poolingPadding.append(self.__formatPad(poolingPad))

        return outputShape, padding, outShapeBeforePool, poolingPadding

    def __findOutPad(self, inputShape, stride, krShape, correlationType):
        if correlationType == self.CorrelationType.VALID:
            outputShape = np.ceil((inputShape[1:] - np.array(krShape[1:])) / stride)
        elif correlationType == self.CorrelationType.FULL:
            outputShape = np.ceil((inputShape[1:] - np.array(krShape[1:]) + (krShape[1:] - np.int16(1)) * 2) / stride)
        elif correlationType == self.CorrelationType.SAME:
            outputShape = np.array(inputShape[1:]) - 1
        else:
            raise ValueError("non valid correlationType")
        outputShape += 1
        pad = self.__findPadFromOut(inputShape, outputShape, stride, krShape)

        return outputShape.astype(np.int16), pad

    @staticmethod
    def __findPadFromOut(inputShape, outputShape, stride, krShape):
        return ((outputShape - np.int16(1)) * stride + krShape[1:] - inputShape[1:]).astype(np.int16)

    @staticmethod
    def __formatStrides(strides, shape):
        newStrides = [np.NAN]
        if iterable(strides):
            for stride in strides:
                if not iterable(stride): stride = stride, stride
                newStrides.append(stride)
        else:
            return ConvolutionalNN.__formatStrides([strides for _ in range(shape.LAYERS - 1)], shape)

        return tuple(newStrides)

    @staticmethod
    def __formatPad(pad):
        return ((x := pad[0] // 2), pad[0] - x), ((y := pad[1] // 2), pad[1] - y)
