from NeuralNetworks import *
from Topologies import *
from Utils import *
from DataSets import dataSet

db = DataBase.load(dataSet.TrainSets.EmnistBalanced)
nn = LoadNeuralNetwork('test.nn.27c.6e.43s.nnt')
