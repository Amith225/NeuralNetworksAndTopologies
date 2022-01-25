from NeuralNetworks import *
from Topologies import *
from Utils import *
from DataSets import dataSet

shape = Shape(10, 20, 30)
print([x.shape for x in Uniform()([(shape[i][0], shape[i - 1][0]) for i in range(shape.LAYERS)])])
