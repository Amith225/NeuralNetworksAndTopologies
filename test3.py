from Topologies import *

aF = activationFuntion

af = aF.Softmax()

aF._np.random.seed(1)
x = aF._np.random.random([2, 3, 2])
ax = af.activation(x)
dx = af.activatedDerivative(ax)

