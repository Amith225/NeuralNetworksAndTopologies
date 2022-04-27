from __init__ import *
from DataSets import dataSet
from Models import model

db = DataBase.load(dataSet.TrainSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1),
                   name='TrainSets.EmnistBalanced')
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1),
                    name='TestSets.EmnistBalanced')
# db2 = False
dense_nn = Dense.NN(shape=Dense.Shape(db.inpShape[0], *(392, 196), db.tarShape[0]),
                    initializers=None,
                    activators=None,
                    lossFunction=None)
dense_nn.train(epochs=5,
               batchSize=128,
               trainDataBase=db,
               optimizers=Optimizers(Optimizers.AdaGrad(), ...),
               profile=False,
               test=db2)
print(db, '\n')
print(db2, '\n')
print(dense_nn, '\n')
