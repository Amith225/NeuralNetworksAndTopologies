from Utils import *
from __Future__.CNN import *
from Topologies import *

l = 1
i = (3, 28, 28)
o = 47
cnn = ConvolutionalNeuralNetwork(shape=Shape(i, [(10, 3, 3), (10, 4, 4)], (10, 2, 2)),
                                 annShape=Shape(100, o),
                                 initializer=Normal(),
                                 annInitializer=Xavier(),
                                 activators=Activators(Prelu(), ..., Softmax()),
                                 annActivators=Activators(Prelu(), ..., Softmax()),
                                 costFunction=MeanSquareLossFunction(),
                                 strides=2,
                                 poolingStride=2, poolingShape=Shape((2, 2), (2, 2)),
                                 poolingType=Types(ConvolutionalNeuralNetwork.PoolingType.MAX, ...),
                                 poolingCorrelationType=Types(ConvolutionalNeuralNetwork.CorrelationType.VALID, ...),
                                 correlationType=Types(ConvolutionalNeuralNetwork.CorrelationType.VALID, ...))

cnn.outputs[0] = np.random.random((l, *i))
cnn._forwardPass()
