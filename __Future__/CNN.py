import typing as tp
if tp.TYPE_CHECKING:
    from ..NeuralNetworks import _
    from ..Topologies import LossFunction, Initializer
    from ..Utils import Activators, Types

import numpy as np

from NeuralNetworks import AbstractNeuralNetwork, ArtificialNeuralNetwork
from Utils import iterable, Shape


class ConvolutionalNeuralNetwork(AbstractNeuralNetwork):
    def _forwardPass(self, layer=1):
        self._fire(layer)
        if layer < self.shape.LAYERS - 1:
            self._forwardPass(layer + 1)
        else:
            # ann here
            pass

    def _backPropagate(self, layer=-1):
        pass

    def _fire(self, layer):
        ###
        if not isinstance(self.outputs[layer-1], int):
            for o in self.outputs[layer-1]:
                self.crossCorrelate(o, layer)
        else: pass

    def _wire(self, layer):
        pass

    def _initializeVars(self):
        self.outputs, self.target = list(range(self.shape.LAYERS)), None
        self.deltaBiases, self.deltaKernels = [np.zeros_like(bias) for bias in self.biasesList], \
                                              [np.zeros_like(kernel) if type(kernel) == np.ndarray else
                                               [np.zeros_like(k) for k in kernel] for kernel in self.kernelsList]

    def _trainer(self):
        """assign self.loss and self.deltaLoss here"""

    class PoolingType:
        MAX = 0; MEAN = 1

    class CorrelationType:
        VALID = 0; FULL = 1; SAME = 2

    def __init__(self, shape: "Shape",
                 annShape: "Shape",
                 initializer: "Initializer",
                 annInitializer: "Initializer",
                 activators: "Activators",
                 annActivators: "Activators",
                 costFunction: "LossFunction",
                 strides,
                 poolingStride, poolingShape,
                 poolingType: "Types",
                 poolingCorrelationType: "Types",
                 correlationType: "Types"):
        super(ConvolutionalNeuralNetwork, self).__init__(shape, activators, costFunction)
        self.strides = self.__formatStrides(strides)
        self.poolingStride = self.__formatStrides(poolingStride)
        self.poolingShape = poolingShape
        self.poolingType = poolingType(self.shape.LAYERS - 1)
        self.poolingCorrelationType = poolingCorrelationType(self.shape.LAYERS - 1)
        self.correlationType = correlationType(self.shape.LAYERS - 1)

        self.outputShape, self.padding, self.outputShapeBeforePool, self.poolingPadding = self.__findOutputShapePadding()
        kernelsShapeFlat = [(1, *self.shape[0])]
        for i, s in enumerate(self.shape[1:]):
            i += 1
            if not iterable(s[0]): kernelsShapeFlat.append((s[0], self.outputShape[i - 1][0], *s[1:]))
            else: kernelsShapeFlat.extend([(ss[0], self.outputShape[i - 1][0], *ss[1:]) for ss in s])
        kernelsListFlat = initializer(kernelsShapeFlat)
        x = 0
        self.kernelsList = [[kernelsListFlat[i + x]] if not iterable(s[0]) else
                            [kernelsListFlat[i + (x := x + y)] for y in range(len(s))]
                            for i, s in enumerate(self.shape)]
        self.biasesList = initializer(self.outputShape)

        annShape = Shape(self.outputShape[-1][0] * self.outputShape[-1][1], *annShape)
        self.ann = ArtificialNeuralNetwork(annShape, annInitializer, annActivators, costFunction)

        self._initializeVars()

    # todo: check if works in this version, also re-learn why this works kek
    def crossCorrelate(self, array, layer):
        kernel = self.kernelsList[layer]
        for j, k in enumerate(kernel):
            paddedArray = np.pad(array, ((0, 0), *self.padding[layer][j]))
            shape = k.shape[1], *self.outputShapeBeforePool[layer][1:], *k.shape[2:]
            stride = paddedArray.strides[0], *[self.strides[layer][i] * paddedArray.strides[i] for i in (-2, -1)],\
                     *paddedArray.strides[-2:]
            crossed = np.lib.stride_tricks.as_strided(paddedArray, shape, stride)
        # return (crossed * kernel).sum(axis=(2, 3))

    # todo: optimize and make more readable, and damn checking is so long
    def __findOutputShapePadding(self):
        outputShape = [self.shape[0]]
        outShapeBeforePool = [self.shape[0]]
        padding = [None]
        poolingPadding = [None]
        for i, krShape in enumerate(self.shape[1:]):
            i += 1
            if iterable(krShape[0]):
                subShape = []
                subPad = []
                depth = 0
                for j, subKrShape in enumerate(krShape):
                    outShape, pad = self.__findOutPad(outputShape[-1], self.strides[i], subKrShape, self.correlationType[i])
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
            raise IOError()
        outputShape += 1
        pad = self.__findPadFromOut(inputShape, outputShape, stride, krShape)

        return outputShape.astype(np.int16), pad

    @staticmethod
    def __findPadFromOut(inputShape, outputShape, stride, krShape):
        return ((outputShape - np.int16(1)) * stride + krShape[1:] - inputShape[1:]).astype(np.int16)

    def __formatStrides(self, strides):
        newStrides = [None]
        if iterable(strides):
            for stride in strides:
                if not iterable(stride):
                    stride = stride, stride
                newStrides.append(stride)
        else:
            return self.__formatStrides([strides for _ in range(self.shape.LAYERS - 1)])

        return tuple(newStrides)

    @staticmethod
    def __formatPad(pad):
        return ((x := pad[0] // 2), pad[0] - x), ((y := pad[1] // 2), pad[1] - y)
