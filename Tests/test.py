from src import *
import pickle

coo = Dense.Layer(shape=Dense.Shape(100, 200, 300),
                  initializer=Initializers.Xavier(5),
                  optimizer=Optimizers.Adam(epsilon=2),
                  activationFunction=Activators.PRelu())
print(coo.__save__()[2]["shape"].save)
