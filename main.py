# todo: dynamic optimizers
# todo: DataBase shaping using self.SHAPE
# todo: auto hyperparameter tuning: Grid search, Population-based natural selection
# todo: auto train stop, inf train
# todo: database save inputs, targets, labels separately
# todo: string hyperparams
# todo: look into "NamedTuple"
# fixme: in DunderSaveLoad for __dict__ saves save only req stuff, maybe by subtracting the vars of base __dict__
import pickle

from src import *
from DataSets import dataSet
from Models import model

db = DataBase.load(dataSet.TrainSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1),
                   name='TrainSets.EmnistBalanced')
db2 = DataBase.load(dataSet.TestSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1),
                    name='TestSets.EmnistBalanced')
# db2 = False
# dense_nn = Dense.NN(shape=Dense.Shape(db.inpShape[0], *(392, 196), db.tarShape[0]),
#                     initializers=None,
#                     activators=None,
#                     lossFunction=None)
# dense_nn.train(epochs=1,
#                batchSize=256,
#                trainDataBase=db,
#                optimizers=None,
#                profile=False,
#                test=db2)
# # print(dense_nn)
#
# coo = dense_nn
# print(coo, id(coo), sep='\n')
# save = coo.__save__()
# with open('temp.save', 'wb') as f:
#     pickle.dump(save, f)
with open('temp.save', 'rb') as f:
    save2 = pickle.load(f)
print(coo2 := load(*save2), id(coo2), sep='\n')
