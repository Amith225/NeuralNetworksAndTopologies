import numpy as np

from __init__ import *

n = 1_000_000
layer = Dense.Layer(Dense.Shape(10, 20), Initializer.Uniform(), Optimizer.GradientDecent())
print(layer)

# layer.forPass(np.random.random((n, *layer.SHAPE.INPUT)))
# layer.backProp(np.random.random((n, *layer.SHAPE.OUTPUT)))
# layer.forPass(np.random.random((n, *layer.SHAPE.INPUT)))
# layer.backProp(np.random.random((n, *layer.SHAPE.OUTPUT)))
