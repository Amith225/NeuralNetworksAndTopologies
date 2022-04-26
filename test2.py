from __init__ import *
import numpy as np
import cProfile as cP

l = 1
i = (1, 28, 28)
o = 47
cnn = ConvolutionalNN(shape=Shape(i, [(10, 3, 3), (10, 4, 4)], (10, 2, 2)), annShape=Shape(100, o),
                      initializer=Normal(), annInitializer=Xavier(),
                      activators=Activators(Prelu(), ..., Softmax()), annActivators=Activators(Prelu(), ..., Softmax()),
                      lossFunction=MeanSquareLossFunction(), annLossFunction=MeanSquareLossFunction(),
                      strides=(1, 1),
                      poolingStride=2,
                      poolingShape=Shape((2, 2), (2, 2)),
                      poolingType=Types(ConvolutionalNN.PoolingType.MAX, ...),
                      poolingCorrelationType=Types(ConvolutionalNN.CorrelationType.VALID, ...),
                      correlationType=Types(ConvolutionalNN.CorrelationType.VALID, ...))

# print(cnn.process(np.random.random((1, *i)).astype(np.float32)).shape)
# cP.run("print(cnn.process(np.random.random((l, *i))).astype(np.float32).shape)")
# print(cnn.process(np.random.random((l, *i)).astype(np.float32)).shape)

from __init__.Topologies import optimizer
db = DataBase(np.random.random((l, *i)).astype(np.float32), np.random.randint(0, o + 1, l), oneHotMaxTar=o)
cnn.train(epochs=1,
          batchSize=2,
          trainDataBase=db,
          optimizer=optimizersNew.GradientDecent(cnn), annOptimizer=AdagradWBOptimizer(cnn.ann),
          profile=False,
          test=False)
