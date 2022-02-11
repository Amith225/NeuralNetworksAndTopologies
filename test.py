from DataSets import dataSet
from NeuralNetworks import *
from Topologies import *
from Utils import *
db = DataBase.load(dataSet.TrainSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1))
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1))

hiddenShape = 392, 196
nn = ArtificialNeuralNetwork(shape=Shape(db.inpShape[0], *hiddenShape, db.tarShape[0]),
                             initializer=Normal(),
                             activators=Activators(Prelu(),
                                                   ...,
                                                   Softmax()),
                             costFunction=MeanSquareLossFunction())

nn.train(2, 64,
         trainDataBase=db,
         optimizer=GradientDecentWBOptimizer(nn, 0.001),
         profile=False,
         test=db2)

# nn.save(replace=True)
# PlotNeuralNetwork.plotCostGraph(nn)
