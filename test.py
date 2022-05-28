from src import *
from DataSets import dataSet
from Models import model

af1 = Optimizers.AdaGrad()
print(af1)
print(id(af1))
print(af2 := af1.__load__(*af1.__save__()))
print(id(af2))
