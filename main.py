# todo: dynamic optimizers
# todo: DataBase shaping using self.SHAPE
# todo: auto hyperparameter tuning: Grid search, Population-based natural selection
# todo: auto train stop, inf train
# todo: database save inputs, targets, labels separately
# todo: string hyperparams
# todo: look into "NamedTuple"
# fixme: in DunderSaveLoad for __dict__ saves save only req stuff, maybe by subtracting the vars of base __dict__
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
# dense_nn.train(epochs=1,
#                batchSize=256,
#                trainDataBase=db,
#                optimizers=None,
#                profile=False,
#                test=db2)
# print(dense_nn)


def save():
    import pickle as dill
    dbb, dbb2 = dense_nn.trainDataBase, dense_nn.testDataBase
    dense_nn.trainDataBase, dense_nn.testDataBase = None, None
    dill.dump(dense_nn, open('t1.nntp', 'wb'))
    dense_nn.trainDataBase, dense_nn.testDataBase = dbb, dbb2


def load():
    import pickle as dill
    return dill.load(open('t1.nntp', 'rb'))
