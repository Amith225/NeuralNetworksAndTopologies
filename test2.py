from NeuralNetworks import *
from Topologies import *
from DataSets import dataSet

db: "DataBase" = DataBase.load(dataSet.TestSets.EmnistBalanced, None, (-1, 1))
PlotDataBase.plotMultiHeight(db.targetSet[:10])
PlotDataBase.show()
