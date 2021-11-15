from DataSets import dataSet
from NeuralNetworks import *
from Topologies import *

AF = WBActivationFunction
nn = ArtificialNeuralNetwork((3, 50, 25, 10, 1),
                             WBInitializer.xavier(2),
                             [AF.elu(), AF.elu(), AF.elu(), AF.elu()])

db = DataBase(*data_set.TrainSets.train_set_xor)
db.normalize()
db2 = DataBase(*data_set.TestSets.test_set_xor)
db2.normalize()

nn.train(10000, 1,
         trainDatabase=db,
         lossFunction=LossFunction.meanSquare(),
         optimizer=WBOptimizer.gradientDecent(nn),
         profile=0)
SaveNeuralNetwork.save(nn)

# out = nn.process(db2.inputSet)
# # print([out])
# tar = db2.targetSet
# out_i = np.where(out == np.max(out, axis=1, keepdims=True))[1]
# tar_i = np.where(tar == 1)[1]
# result = out_i == tar_i
# result = np.array(1) * result
# print(result.sum() / len(result) * 100)
#
# out = nn.process(db.inputSet)
# # print([out])
# tar = db.targetSet
# out_i = np.where(out == np.max(out, axis=1, keepdims=True))[1]
# tar_i = np.where(tar == 1)[1]
# result = out_i == tar_i
# result = np.array(1) * result
# print(result.sum() / len(result) * 100)

PlotNeuralNetwork().plotCostGraph(nn)
