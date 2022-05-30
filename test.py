from src import *

coo = Dense.Layer(shape=Dense.Shape(784, 398),
                  initializer=Initializers.Xavier(5),
                  optimizer=Optimizers.Adam(epsilon=2),
                  activationFunction=Activators.PRelu())
print(coo, id(coo))
print(coo2 := load(*coo.__save__()), id(coo2))
