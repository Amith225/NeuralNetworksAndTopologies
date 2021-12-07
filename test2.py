from NeuralNetworks import *
from Topologies import *
from DataSets import dataSet

db: "DataBase" = DataBase.load(dataSet.TrainSets.EmnistBalanced, hotEncodeTar=1)
