from NeuralNetworks import *
from Topologies import *
from DataSets import dataSet

db: "DataBase" = DataBase.load(dataSet.TestSets.EmnistBalanced, None, (-1, 1))
PlotDataBase.plotMultiMap(db.inputSet[:100].reshape((-1, 28, 28)))
PlotDataBase.show()
