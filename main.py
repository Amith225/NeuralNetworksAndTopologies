# todo: dynamic optimizers
# todo: DataBase shaping using self.SHAPE
# todo: auto hyperparameter tuning: Grid search, Population-based natural selection
# todo: auto train stop, inf train
# todo: database save inputs, targets, labels separately
from src import *
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
dense_nn.train(epochs=10,
               batchSize=256,
               trainDataBase=db,
               optimizers=None,
               profile=False,
               test=db2)
print(db, db2, dense_nn, sep='\n\n')
