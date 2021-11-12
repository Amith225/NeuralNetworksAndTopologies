import numpy as np


class LossFunction:
    def __init__(self, lossFunction):
        self.__eval = lossFunction

    def eval(self, output, target):
        return self.__eval(output, target)

    @staticmethod
    def meanSquare():
        def lossFunction(output, target):
            loss = output - target

            return np.einsum('lij,lij->', loss, loss), loss

        return LossFunction(lossFunction)
