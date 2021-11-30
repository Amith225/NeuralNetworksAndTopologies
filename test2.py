from Topologies.dataBase import DataBase
from DataSets import dataSet
import numpy as np

db = DataBase.load(dataSet.TrainSets.EmnistBalanced, 1)
db.save(replace=True)
db = DataBase.load(dataSet.TrainSets.EmnistBalanced)
db2 = DataBase.load('db112800s784i47o.npzdb')

print(np.alltrue(db.inputSet == db2.inputSet), np.alltrue(db.targetSet == db2.targetSet))
