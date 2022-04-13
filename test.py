from DataSets import dataSet
from __init__ import *

db = DataBase.load(dataSet.TrainSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1))
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1))

hiddenShape = 392, 196
nn = ArtificialNN(shape=Shape(db.inpShape[0], *hiddenShape, db.tarShape[0]),
                  initializer=Xavier(2),
                  activators=Activators(Prelu(),
                                        ...,
                                        Softmax()),
                  lossFunction=MeanSquareLossFunction())

nn.train(2, 64,
         trainDataBase=db,
         optimizer=AdagradWBOptimizer(nn),
         profile=False,
         test=db2)

# nn.save(replace=True)
# PlotNeuralNetwork.plotCostGraph(nn)
