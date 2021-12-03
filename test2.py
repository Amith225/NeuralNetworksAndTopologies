from NeuralNetworks import AbstractNeuralNetwork, ArtificialNeuralNetwork, LoadNeuralNetwork, PlotNeuralNetwork
from Topologies import DataBase
from DataSets import dataSet
from utils import Plot
import numpy as np

db: "DataBase" = DataBase.load('db112800s28i47o.zdb')
# db = DataBase(db.inputSet.astype(np.uint8).reshape((-1, 28, 28)), db.targetSet.astype(np.uint8))
# db.save()
Plot.plotInputVecAsImg(db.inputSet[:30].transpose([0, 2, 1]))
# nn = LoadNeuralNetwork('nn0.24c4e01m06s.nnt')
# PlotNeuralNetwork.plotCostGraph(nn)
Plot.show()
