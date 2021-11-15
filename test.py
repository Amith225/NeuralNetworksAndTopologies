from DataSets import dataSet
from NeuralNetworks import *
from Topologies import *

import numpy as np

aF = WBActivationFunction
nn = ArtificialNeuralNetwork(wbShape=WBShape(784, 392, 196, 47),
                             wbInitializer=WBInitializer.xavier(2),
                             wbActivations=Activations(aF.prelu(), ..., aF.softmax()))

db = DataBase.load(dataSet.TrainSets.EmnistBalanced)
db.normalize()
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced)
db2.normalize()

nn.train(3, 64,
         trainDatabase=db,
         lossFunction=LossFunction.meanSquare(),
         wbOptimizer=NesterovMomentumWBOptimizer(nn, 0.001),
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
