from DataSets import dataSet
from NeuralNetworks import *
from Topologies import *
from utils import *

db = DataBase.load(dataSet.TrainSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1))
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1))

hiddenShape = 392, 196
nn = ArtificialNeuralNetwork(wbShape=WBShape(db.inpShape[0], *hiddenShape, db.tarShape[0]),
                             wbInitializer=XavierWBInitializer(2),
                             activators=Activators(Prelu(),
                                                   ...,
                                                   Softmax()))

nn.train(2, 256,
         trainDataBase=db2,
         costFunction=MeanSquareLossFunction(),
         wbOptimizer=AdagradWBOptimizer(nn),
         profile=False,
         test=db2)

# nn.save(replace=True)
# PlotNeuralNetwork.plotCostGraph(nn)
