from DataSets import data_set
from NeuralNetworks import *
from Topologies import *

import numpy as np

aF = WBActivationFunction
nn = ArtificialNeuralNetwork(wbShape=WBShape(784, 392, 196, 47),
                             wbInitializer=WBInitializer.xavier(2),
                             wbActivations=Activations(aF.relu(), ..., aF.softmax()))

db = DataBase(*data_set.TRAIN_SETS.train_set_emnist_balanced)
db.normalize()
db2 = DataBase(*data_set.TEST_SETS.test_set_emnist_balanced)
db2.normalize()

nn.train(2, 64,
         trainDatabase=db,
         lossFunction=LossFunction.meanSquare(),
         wbOptimizer=WBOptimizer.nesterovMomentum(nn, 0.001, 0.0001),
         profile=False)

# out = nn.process(db2.inputSet)
# tar = db2.targetSet
# outIndex = np.where(out == np.max(out, axis=1, keepdims=True))[1]
# targetIndex = np.where(tar == 1)[1]
# result = outIndex == targetIndex
# result = np.array(1) * result
# print(result.sum() / len(result) * 100)

# out = nn.process(db.inputSet)
# print([out])
# tar = db.targetSet
# outIndex = np.where(out == np.max(out, axis=1, keepdims=True))[1]
# targetIndex = np.where(tar == 1)[1]
# result = outIndex == targetIndex
# result = np.array(1) * result
# print(result.sum() / len(result) * 100)

# PlotNeuralNetwork.plotCostGraph(nn)

# while 1:
#     inp = input("('e' exit)>>> ")
#     if inp.lower() == 'e':
#         break
#     print(round(nn.process(eval(inp))[0][0][0], 2))
