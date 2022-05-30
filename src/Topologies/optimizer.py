from typing import Callable
from abc import ABCMeta, abstractmethod

import numpy as np
import numexpr as ne

from ..tools import DunderSaveLoad, load


class BaseOptimizer(DunderSaveLoad, metaclass=ABCMeta):
    ZERO, ONE = np.float32(0), np.float32(1)
    _dict = True

    def __repr__(self):
        lr = self.LEARNING_RATE
        return f"<{self.__class__.__name__}:{lr=}>"

    def __new_copy__(self):
        return load(*self.__save__())

    def __init__(self, learningRate: float):
        self.LEARNING_RATE = np.float32(learningRate)

    def __call__(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray"):
        return self._optimize(grad, theta)

    @abstractmethod
    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        pass


class GradientDecent(BaseOptimizer):
    def __init__(self, learningRate: float = None):
        if learningRate is None: learningRate = .001
        super(GradientDecent, self).__init__(learningRate)

    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        delta = grad(theta)
        (local_dict := vars(self)).update(locals())
        return ne.evaluate("delta * LEARNING_RATE", local_dict=local_dict)


class Decay(BaseOptimizer):
    def __repr__(self):
        decay = self.DECAY
        return f"{super(Decay, self).__repr__()[:-1]}: {decay=}>"

    def __init__(self, learningRate: float = None, decay: float = None):
        if learningRate is None: learningRate = .001
        super(Decay, self).__init__(learningRate)
        if decay is None: decay = self.LEARNING_RATE ** 2
        self.DECAY = np.float32(decay)
        self.decayCounter = self.ZERO

    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        delta = grad(theta)
        self.decayCounter += self.ONE
        locals()['ONE'] = self.ONE
        (local_dict := vars(self)).update(locals())
        return ne.evaluate("delta * LEARNING_RATE / (ONE + decayCounter * DECAY)", local_dict=local_dict)


class Momentum(BaseOptimizer):
    def __repr__(self):
        moment = self.MOMENT
        return f"{super(Momentum, self).__repr__()[:-1]}: {moment=}>"

    def __init__(self, learningRate: float = None, moment: float = None):
        if learningRate is None: learningRate = .001
        super(Momentum, self).__init__(learningRate)
        if moment is None: moment = .5
        self.MOMENT = np.float32(moment)
        self.prevDelta = self.ZERO

    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        delta = grad(theta)
        (local_dict := vars(self)).update(locals())
        self.prevDelta = momentDelta = ne.evaluate("LEARNING_RATE * delta + MOMENT * prevDelta", local_dict=local_dict)
        return momentDelta


class NesterovMomentum(BaseOptimizer):
    def __repr__(self):
        moment = self.MOMENT
        return f"{super(NesterovMomentum, self).__repr__()[:-1]}: {moment=}>"

    def __init__(self, learningRate: float = None, moment: float = None):
        if learningRate is None: learningRate = .001
        super(NesterovMomentum, self).__init__(learningRate)
        if moment is None: moment = .5
        self.MOMENT = np.float32(moment)
        self.prevDelta = self.ZERO

    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        delta = grad(theta - self.MOMENT * self.prevDelta)
        (local_dict := vars(self)).update(locals())
        self.prevDelta = momentDelta = ne.evaluate("LEARNING_RATE * delta + MOMENT * prevDelta", local_dict=local_dict)
        return momentDelta


class AdaGrad(BaseOptimizer):
    def __repr__(self):
        eps = self.EPSILON
        return f"{super(AdaGrad, self).__repr__()[:-1]}: {eps=}>"

    def __init__(self, learningRate: float = None, epsilon: float = None):
        if learningRate is None: learningRate = .01
        super(AdaGrad, self).__init__(learningRate)
        if epsilon is None: epsilon = 1e-7
        self.EPSILON = np.float32(epsilon)
        self.summationSquareDelta = self.ZERO

    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        delta = grad(theta)
        (local_dict := vars(self)).update(locals())
        self.summationSquareDelta = ne.evaluate('summationSquareDelta + delta * delta', local_dict=local_dict)
        (local_dict := vars(self)).update(locals())
        return ne.evaluate('delta * LEARNING_RATE / sqrt(summationSquareDelta + EPSILON)', global_dict=local_dict)


class RmsProp:
    def __init__(self):
        raise NotImplementedError


class AdaDelta:
    def __init__(self):
        raise NotImplementedError


class Adam(BaseOptimizer):
    def __repr__(self):
        b1, b2 = self.BETA1, self.BETA2
        eps = self.EPSILON
        return f"{super(Adam, self).__repr__()[:-1]}: {b1=}: {b2=}: {eps=}>"

    def __init__(self, learningRate: float = None, beta1: float = None, beta2: float = None, epsilon: float = None,
                 decay: float = None):
        if learningRate is None: learningRate = .001
        super(Adam, self).__init__(learningRate)
        if beta1 is None: beta1 = .9
        self.BETA1 = np.float32(beta1)
        self.BETA1_BAR = 1 - self.BETA1
        if beta2 is None: beta2 = .999
        self.BETA2 = np.float32(beta2)
        self.BETA2_BAR = 1 - self.BETA2
        if epsilon is None: epsilon = 1e-7
        self.EPSILON = np.float32(epsilon)
        if decay is None: decay = NotImplemented  # todo: implement decay on decayCounter?
        # self.DECAY = np.float32(decay)
        self.decayCounter = self.ONE
        self.weightedSummationDelta = self.ZERO
        self.weightedSummationSquareDelta = self.ZERO

    def _optimize(self, grad: Callable[["np.ndarray"], "np.ndarray"], theta: "np.ndarray") -> "np.ndarray":
        delta = grad(theta)
        (local_dict := vars(self)).update(locals())
        self.weightedSummationDelta = ne.evaluate(
            "BETA1 * weightedSummationDelta + BETA1_BAR * delta", local_dict=local_dict)
        self.weightedSummationSquareDelta = ne.evaluate(
            "BETA2 * weightedSummationSquareDelta + BETA2_BAR * delta * delta", local_dict=local_dict)

        (local_dict := vars(self)).update(locals())
        weightedSummationDeltaHat = ne.evaluate(
            "weightedSummationDelta / (1 - BETA1 ** decayCounter)", local_dict=local_dict)
        weightedSummationSquareDeltaHat = ne.evaluate(
            "weightedSummationSquareDelta / (1 - BETA2 ** decayCounter)", local_dict=local_dict)

        self.decayCounter += self.ONE
        (local_dict := vars(self)).update(locals())
        return ne.evaluate(
            "LEARNING_RATE * weightedSummationDeltaHat / sqrt(weightedSummationSquareDeltaHat + EPSILON)",
            local_dict=local_dict)
