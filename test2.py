from Utils import *
from __Future__.CNN import *
from Topologies import *

cnn = ConvolutionalNeuralNetwork(shape=Shape((28, 28, 3), [(3, 3, 10), (4, 4, 10)], (3, 3, 10)),
                                 annShape=Shape(100, 47),
                                 initializer=Normal(),
                                 annInitializer=Xavier(),
                                 activators=Activators(Prelu(), ..., Softmax()),
                                 annActivators=Activators(Prelu(), ..., Softmax()),
                                 costFunction=MeanSquareLossFunction(),
                                 strides=[2, 1],
                                 paddingVal=[0, 0],
                                 poolingStride=[1, 1], poolingShape=Shape((2, 2), (2, 2)),
                                 poolingType=ConvolutionalNeuralNetwork.PoolingType.MAX,
                                 correlationType=ConvolutionalNeuralNetwork.CorrelationType.FULL)
print(cnn.outputShape)
