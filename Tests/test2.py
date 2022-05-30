from src import *
from DataSets import dataSet
from Models import model

cl = Conv.Layer(UniversalShape((3, 28, 28), (10, 3, 3), (5, 4, 4)),
                Initializers.Xavier(),
                Optimizers.AdaGrad(),
                Activators.PRelu())

print(cl)
