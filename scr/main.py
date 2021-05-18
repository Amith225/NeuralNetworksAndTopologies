import numpy as np

import Neural_Network as Nn
import Topology as Tp

shape = (784, 100, 10)
initializer = Tp.Initializer.normal()
activation = Tp.ActivationFunction.sigmoid()
output_activation = Tp.ActivationFunction.sigmoid()

dense = Nn.CreateNeuralNetwork(shape=shape,
                               initializer=initializer,
                               activation=activation,
                               output_activation=output_activation)


# np_loader = np.load('image_classification_47_balanced_test.npz')
# data_base = Tp.CreateDatabase(np_loader['arr_0']/256, np_loader['arr_1'])
n = 10000
data_base = Tp.CreateDatabase(Nn.np.random.random([n, shape[0]]), Nn.np.random.random([n, shape[-1]]))
epochs = 10
batch_size = -1
loss_function = Tp.LossFunction.mean_square()
optimizer = Tp.Optimizer.traditional_gradient_decent(dense, 1)

dense.train(train_database=data_base,
            epochs=epochs,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer=optimizer)
