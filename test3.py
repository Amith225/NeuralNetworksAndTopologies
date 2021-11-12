from DataSets import data_set
from NeuralNetworks import *
from Topologies import *

aF = WBActivationFunction
nn = ArtificialNeuralNetwork(wbShape=WBShape(3, 10, 10, 1),
                             wbInitializer=WBInitializer.xavier(2),
                             wbActivations=Activations(aF.relu(), ..., aF.sigmoid()))

db = DataBase(*data_set.TRAIN_SETS.train_set_xor)
db.normalize()
db2 = DataBase(*data_set.TEST_SETS.test_set_xor)
db2.normalize()

nn.train(10000, 2,
         trainDatabase=db,
         lossFunction=LossFunction.meanSquare(),
         optimizer=Optimizer.gradientDecent(nn),
         profile=False)
