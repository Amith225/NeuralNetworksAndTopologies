from DataSets import dataSet
from NeuralNetworks import *
from Topologies import *

import numpy as np

# xor = [[[0], [0], [0]], [[0], [0], [1]], [[0], [1], [0]], [[0], [1], [1]],
#        [[1], [0], [0]], [[1], [0], [1]], [[1], [1], [0]], [[1], [1], [1]]], \
#       [[[0]], [[1]], [[1]], [[0]], [[1]], [[0]], [[0]], [[1]]]
db = DataBase.load(dataSet.TrainSets.EmnistBalanced)
db.normalize()
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced)
db2.normalize()

aF = WBActivationFunction
hiddenShape = 392, 196
nn = ArtificialNeuralNetwork(wbShape=WBShape(db.inpShape, *hiddenShape, db.tarShape),
                             wbInitializer=XavierWBInitializer(2),
                             wbActivations=Activations(aF.prelu(), ..., aF.softmax()))

nn.train(4, 128,
         trainDatabase=db,
         lossFunction=LossFunction.meanSquare(),
         wbOptimizer=AdagradWBOptimizer(nn),
         profile=False)


def accuracy(nn, db):
    out = nn.process(db.inputSet)
    tar = db.targetSet
    outIndex = np.where(out == np.max(out, axis=1, keepdims=True))[1]
    targetIndex = np.where(tar == 1)[1]
    result = outIndex == targetIndex
    result = np.array(1) * result

    return result.sum() / len(result) * 100


# print('Train Accuracy: ', accuracy(nn, db))
# print('Test Accuracy: ', accuracy(nn, db2))
# PlotNeuralNetwork.plotCostGraph(nn)
