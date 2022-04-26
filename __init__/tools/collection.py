from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..NeuralNetworks import *
    from ..Topologies import *


class Collections:
    def __init__(self, *collectables):
        self.collectables = collectables

    def __call__(self, length):
        return self.get(length)

    def get(self, length):
        trueCollectables = []
        prevCollectable = None
        numEllipsis = self.collectables.count(Ellipsis)
        numCollectables = len(self.collectables) - numEllipsis
        vacancy = length - numCollectables
        for collectable in self.collectables:
            if collectable == Ellipsis:
                for i in range(filled := (vacancy // numEllipsis)):
                    trueCollectables.append(prevCollectable)
                vacancy -= filled
                numEllipsis -= 1
                continue
            trueCollectables.append(collectable)
            prevCollectable = collectable

        return trueCollectables


class Activators(Collections):
    def __init__(self, *activationFunctions: "ActivationFunction.Abstract"):
        super(Activators, self).__init__(*activationFunctions)

    def __call__(self, length):
        activations = []
        activationDerivatives = []
        for e in self.get(length):
            activations.append(e.activation)
            activationDerivatives.append(e.activatedDerivative)

        return activations, activationDerivatives


class Initializers(Collections):
    def __init__(self, *initializer: "Initializer.Base"):
        super(Initializers, self).__init__(*initializer)


class Optimizers(Collections):
    def __init__(self, *optimizers: "Optimizer.Base"):
        super(Optimizers, self).__init__(*optimizers)


class PoolingTypes(Collections):
    def __init__(self, *types: "ConvolutionalNN.PoolingType"):
        super(PoolingTypes, self).__init__(*types)


class CorrelationTypes(Collections):
    def __init__(self, *types: "ConvolutionalNN.CorrelationType"):
        super(CorrelationTypes, self).__init__(*types)
