import numpy as np

import Neural_Network as Nn
import Topology as Tp

shape = (784, 56, 47)
initializer = None
activation = None
output_activation = None

dense = Nn.CreateNeuralNetwork(shape=shape,
                               initializer=initializer,
                               activation=activation,
                               output_activation=output_activation)


np_loader = np.load('image_classification_47_balanced_test.npz')
data_base = Tp.CreateDatabase(np_loader['arr_0']/256, np_loader['arr_1'])
epochs = 100
batch_size = None
loss_function = None
optimizer = None

dense.train(train_database=data_base,
            epochs=epochs,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer)
