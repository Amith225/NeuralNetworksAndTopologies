from NeuralNetworks import AbstractNeuralNetwork, ArtificialNeuralNetwork, LoadNeuralNetwork, PlotNeuralNetwork
from Topologies import DataBase, PlotDataBase
from DataSets import dataSet
import numpy as np

db: "DataBase" = DataBase.load(dataSet.TestSets.EmnistBalanced)
PlotDataBase.plotInputSet(db, 10, 100)
