import typing as tp
if tp.TYPE_CHECKING:
    from ..NeuralNetworks import _
    from ..Topologies import LossFunction, Initializer
    from ..Utils import Activators

import numpy as np

from NeuralNetworks import AbstractNeuralNetwork, ArtificialNeuralNetwork
from Utils import iterable, Shape


class ConvolutionalNeuralNetwork(AbstractNeuralNetwork):
    def _forwardPass(self, layer=1):
        pass

    def _backPropagate(self, layer=-1):
        pass

    def _fire(self, layer):
        pass

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
                 paddingVal,
                 poolingStride, poolingShape,
                 poolingType: "ConvolutionalNeuralNetwork.PoolingType",
                 correlationType: "ConvolutionalNeuralNetwork.CorrelationType"):
        self.strides = self.__formatStrides(strides)
        self.paddingVal = paddingVal
        self.poolingStride = self.__formatStrides(poolingStride)
        self.poolingShape = poolingShape
        self.poolingType = poolingType
        self.correlationType = correlationType

        super(ConvolutionalNeuralNetwork, self).__init__(shape, activators, costFunction)

        self.outputShape, self.padding = self.__findOutputShapePadding()
        kernelsShapeFlat = []
        for s in self.shape:
            if not iterable(s[0]): kernelsShapeFlat.append(s)
            else: kernelsShapeFlat.extend(s)
        kernelsListFlat = initializer(kernelsShapeFlat)
        x = 0
        self.kernelsList = [kernelsListFlat[i + x] if not iterable(s[0]) else
                            [kernelsListFlat[i + (x := x + y)] for y in range(len(s))]
                            for i, s in enumerate(self.shape)]
        self.biasesList = initializer(self.outputShape)

        annShape = Shape(self.outputShape[-1][0] * self.outputShape[-1][1], *annShape.shape)
        self.ann = ArtificialNeuralNetwork(annShape, annInitializer, annActivators, costFunction)

        self._initializeVars()

    def __findOutputShapePadding(self):
        outputShape = [self.shape[0]]
        padding = [None]
        for i, krShape in enumerate(self.shape[1:]):
            i += 1
            if iterable(krShape[0]):
                subShape = []
                subPad = []
                depth = 0
                for j, subKrShape in enumerate(krShape):
                    outShape, pad = self.__findOutPad(outputShape, self.strides[i], subKrShape, self.correlationType)
                    subShape.append((*outShape, subKrShape[2]))
                    subPad.append(pad)
                    depth += subKrShape[2]
                outShape = *np.maximum.reduce(subShape)[:2], depth
                subPad += np.subtract(outShape, subShape)[:, :2] * self.strides[i]
                padding.append([self.__formatPad(e) for e in subPad])
            else:
                outShape, pad = self.__findOutPad(outputShape, self.strides[i], krShape, self.correlationType)
                outShape = *outShape, krShape[2]
                padding.append(self.__formatPad(pad))
            depth = outShape[2]
            outShape, poolingPad = self.__findOutPad([outShape], self.poolingStride[i], self.poolingShape[i - 1],
                                                     ConvolutionalNeuralNetwork.CorrelationType.VALID)
            outputShape.append((*outShape, depth))

        return outputShape, padding

    @staticmethod
    def __findOutPad(outputShape, stride, krShape, correlationType):
        outShape, rem = np.divmod(np.subtract(outputShape[-1][:2], krShape[:2]), stride)
        if correlationType == ConvolutionalNeuralNetwork.CorrelationType.VALID:
            pad = (krShape[:2] - rem) * (rem != 0) + stride
        elif correlationType == ConvolutionalNeuralNetwork.CorrelationType.FULL:
            pad = (krShape[:2] - np.int16(1)) * 2 + (krShape[:2] - rem) * (rem != 0) + stride
        elif correlationType == ConvolutionalNeuralNetwork.CorrelationType.SAME:
            pad = outputShape[-1][:2] * (stride - np.int16(1)) + krShape[:2] - 1
        else:
            raise IOError()
        outShape += np.ceil(pad / stride).astype(int)

        return outShape, pad

    @staticmethod
    def __formatStrides(strides):
        newStrides = [None]
        for stride in strides:
            if not iterable(stride):
                stride = stride, stride
            newStrides.append(stride)

        return tuple(newStrides)

    @staticmethod
    def __formatPad(pad):
        return ((x := pad[0] // 2), pad[0] - x), ((y := pad[1] // 2), pad[1] - y)

    @staticmethod
    def __formatPoolingType(poolingType):
        pass

    @staticmethod
    def __formatCorrelationType(correlationType):
        pass
