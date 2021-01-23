import Neural_Network as Nn
import Topology as Tp

shape = None
initializer = None
activation = None
output_activation = None

dense = Nn.CreateNeuralNetwork(shape=shape,
                               initializer=initializer,
                               activation=activation,
                               output_activation=output_activation)
