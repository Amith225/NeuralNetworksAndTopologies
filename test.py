from DataSets import dataSet
from NeuralNetworks import *
from Topologies import *
from utils import *

db = DataBase.load(dataSet.TrainSets.EmnistBalanced, 1)
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced, 1)

aF = activationFuntion
hiddenShape = 392, 196
nn = ArtificialNeuralNetwork(wbShape=WBShape(db.inpShape, *hiddenShape, db.tarShape),
                             wbInitializer=XavierWBInitializer(2),
                             activators=Activators(aF.Prelu(),
                                                   ...,
                                                   aF.Softmax()))

nn.train(2, 256,
         trainDatabase=db,
         lossFunction=MeanSquareLossFunction(),
         wbOptimizer=AdagradWBOptimizer(nn),
         profile=False,
         test=db2)

nn2 = LoadNeuralNetwork(nn.save())
print(nn.weightsList[0] == nn2.weightsList[0])

# PlotNeuralNetwork.plotCostGraph(nn)
