import time 
import warnings 
import cProfile 
import traceback 
import pstats 
import os 
from abc import ABCMeta, abstractmethod
from typing import Callable, Union, Iterable, Generator, TYPE_CHECKING, Sized
import numpy as np
from matplotlib import widgets as wg, pyplot as plt
import tempfile 
import ctypes 
from numpy.lib import format as fm
import sys 
import inspect 
import numexpr as ne


# /DataSets/dataSet.py
class TrainSets:
    EmnistBalanced = "emnist.balanced.train.112800s.(28,28)i.(47,1)o.zdb"
    Xor = "xor3.train.8s.(3,1)i.(1,1)o.zdb"


class TestSets:
    EmnistBalanced = "emnist.balanced.test.18800s.(28,28)i.(47,1)o.zdb"
    Xor = "xor3.test.8s.(3,1)i.(1,1)o.zdb"

# /Models/model.py

# /__init__/tools/base.py
pass  # import os
pass  # from abc import ABCMeta, abstractmethod

pass  # import numpy as np
pass  # from matplotlib import pyplot as plt, widgets as wg


class BaseSave(metaclass=ABCMeta):
    DEFAULT_DIR: str
    DEFAULT_NAME: str
    FILE_TYPE: str

    @abstractmethod
    def saveName(self) -> str:
        pass

    @abstractmethod
    def save(self, file: str = None, replace: bool = False) -> str:
        if file is None: file = self.DEFAULT_NAME
        if not (fpath := os.path.dirname(file)):
            fpath = f"{os.getcwd()}\\{self.DEFAULT_DIR}\\"
            fName = file
        else:
            fpath += '\\'
            fName = os.path.basename(file)
        os.makedirs(fpath, exist_ok=True)
        if len(fName) >= (typeLen := 1 + len(self.FILE_TYPE)) and fName[1 - typeLen:] == self.FILE_TYPE:
            fName = fName[:-typeLen]
            savePath = f"{fpath}{fName.replace(' ', '_')}"
        else:
            savePath = f"{fpath}{fName.replace(' ', '_')}_{self.saveName().replace(' ', '')}"

        numSavePath = savePath
        if not replace:
            i = 0
            while 1:
                if i != 0: numSavePath = f"{savePath} ({i})"
                if not os.path.exists(f"{numSavePath}.{self.FILE_TYPE}"): break
                i += 1

        dumpFile = f"{numSavePath}.{self.FILE_TYPE}"
        return dumpFile


class BaseLoad(metaclass=ABCMeta):
    DEFAULT_DIR: str
    FILE_TYPE: str

    @classmethod
    @abstractmethod
    def load(cls, file) -> "cls":  # noqa
        if file:
            if not (fpath := os.path.dirname(file)):
                fpath = f"{os.getcwd()}\\{cls.DEFAULT_DIR}\\"
                fName = file
            else:
                fpath += '\\'
                fName = os.path.basename(file)
        else:
            raise NameError("file not given")
        if '.' not in fName: fName += cls.FILE_TYPE

        loadFile = fpath + fName
        return loadFile


plt.style.use('dark_background')
class Plot:  # noqa
    @staticmethod
    def __init_ax(ax):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax, fig = ax, ax.figure

        return ax

    @staticmethod
    def __plotMulti(count, plotter, rows, columns):
        MAX = rows * columns
        fig = plt.figure()
        axes = [fig.add_subplot(rows + 1, columns, i + 1) for i in range(MAX)]
        butPrev = wg.Button(ax := fig.add_subplot(rows + 1, 2, 2 * rows + 1), '<-', color='red', hovercolor='blue')
        ax.set_aspect(.1)
        butNext = wg.Button(ax := fig.add_subplot(rows + 1, 2, 2 * rows + 2), '->', color='red', hovercolor='blue')
        ax.set_aspect(.1)
        fig.page = 0

        def onclick(_page):
            if _page == 0:
                butPrev.active = False
                butPrev.ax.patch.set_visible(False)
            else:
                butPrev.active = True
                butPrev.ax.patch.set_visible(True)
            if _page == count // MAX:
                butNext.active = False
                butNext.ax.patch.set_visible(False)
            else:
                butNext.active = True
                butNext.ax.patch.set_visible(True)
            fig.page = _page
            [(_ax.clear(), _ax.set_xticks([]), _ax.set_yticks([])) for _ax in axes]
            plotter(_page, axes)
            fig.subplots_adjust(.01, .01, .99, .99, 0, 0)
            fig.canvas.draw()

        butNext.on_clicked(lambda *_: onclick(fig.page + 1))
        butPrev.on_clicked(lambda *_: onclick(fig.page - 1))
        onclick(fig.page)

        return axes

    @staticmethod
    def plotHeight(xs, ys=None, cluster=False, join=True,
                   scatter=False, scatterLabels=None, scatterRotation=0, scatterSize=100,
                   text='', textPos=(.01, .01), textC='yellow', ax=None, multi=False, rows=4, columns=4):
        if multi:
            MAX = rows * columns
            length = np.shape(xs)[0]
            if text is None:
                text = range(length)
            if scatterLabels is None:
                scatterLabels = range(length)
            if ys is None:
                ys = [None for _ in range(length)]

            def plotter(_page, axes):
                if (to := (_page + 1) * MAX) > length:
                    to = length
                for ind, x in enumerate(xs[_page * MAX:to]):
                    Plot.plotHeight(x, ys[ind], cluster, join, scatter, scatterLabels, scatterRotation,
                                    text=str(text[_page * MAX + ind]), ax=axes[ind])

            return Plot.__plotMulti(xs.shape[0], plotter, rows, columns)

        ax = Plot.__init_ax(ax)
        args = (xs, ys)
        if ys is None:
            args = (xs,)
        if not cluster:
            args = [[arg] for arg in args]
            scatterLabels = [scatterLabels]
        for i in range(len(xs)):
            points = [arg[i] for arg in args if arg[i] is not None]
            if join:
                ax.plot(*points, c=np.random.rand(3))
            if scatter or scatterLabels is not None:
                ax.scatter(*points, s=scatterSize, color="gray")
                if scatterLabels is not None:
                    for *point, label in zip(*points, scatterLabels[i]):
                        ax.annotate(label, point, rotation=scatterRotation)
        ax.text(*textPos, text, transform=ax.transAxes, c=textC)

        return ax

    @staticmethod
    def plotMap(vect, text=None, textPos=(.01, .01), textC='yellow', ax=None, rows=4, columns=4):
        if len(np.shape(vect)) != 2:
            MAX = rows * columns
            length = np.shape(vect)[0]
            if text is None:
                text = range(length)

            def plotter(_page, axes):
                if (to := (_page + 1) * MAX) > length:
                    to = length
                for i, im in enumerate(vect[_page * MAX:to]):
                    Plot.plotMap(im, text=str(text[_page * MAX + i]), ax=axes[i])

            return Plot.__plotMulti(length, plotter, rows, columns)

        ax = Plot.__init_ax(ax)
        ax.imshow(vect)
        ax.text(*textPos, text, transform=ax.transAxes, c=textC)

        return ax

    @staticmethod
    def show():
        plt.show()

# /__init__/tools/helperClass.py
pass  # import tempfile
pass  # import ctypes
pass  # from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # from ..tools import *

pass  # import numpy as np
pass  # from numpy.lib import format as fm


class NumpyDataCache(np.ndarray):
    def __new__(cls, array):
        return cls.writeNpyCache(array)

    @staticmethod
    def writeNpyCache(array: "np.ndarray") -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix='.npy') as file:
            np.save(file, array)
            file.seek(0)
            fm.read_magic(file)
            fm.read_array_header_1_0(file)
            memMap = np.memmap(file, mode='r', shape=array.shape, dtype=array.dtype, offset=file.tell())

        return memMap


class Collections:
    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.collectables}>"

    # todo: make collectables Type[_<class>] itself, and/or create Collection class generator in general
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


try:
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except:  # noqa
    pass
# noinspection SpellCheckingInspection
class PrintCols:  # noqa
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBOLDITALIC = CBOLD + CITALIC
    CURLBOLD = CBOLD + CURL
    CITALICURL = CITALIC + CURL
    CBOLDITALICURL = CBOLD + CITALIC + CURL

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'

# /__init__/tools/magicProperty.py
pass  # import inspect


class MagicBase:
    def __new__(cls, *args, **kwargs):
        obj = super(MagicBase, cls).__new__(cls)
        obj.toMagicProperty = set()
        return obj

    def __magic_start__(self):
        self.toMagicProperty = set(self.__dict__.keys())

    def __magic_end__(self):
        self.toMagicProperty = set(self.__dict__.keys()) - self.toMagicProperty


class MagicProperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(MagicProperty, self).__init__(fget, self.makeMagicF(fset), self.makeMagicF(fdel), doc)
        self.__obj = self.getCaller()

    def makeMagicF(self, _f):
        def f(*args, **kwargs):
            if self.__magic__():
                return _f(*args, **kwargs)
            else:
                raise AttributeError("Attribute is read only")

        return f

    def __magic__(self, stack=1):
        caller = self.getCaller(stack + 1)
        return any(c1 == c2 and c1 is not None for c1, c2 in zip(caller, self.__obj)) or \
            (any(self.__obj[2] == base.__name__ for base in caller[1].__bases__)
                if self.__obj[:2] == (None, None) and caller[1] is not None else 0)

    @staticmethod
    def getCaller(stack=1):
        caller = (callStack := inspect.stack()[stack + 1][0].f_locals).get('self')
        _return = caller, caller.__class__, caller.__class__.__name__
        if caller is None:
            _return = None, (caller := callStack.get('cls')), caller.__name__ if caller is not None else None
        if caller is None: _return = None, None, callStack.get('__qualname__')
        return _return


def makeMetaMagicProperty(*inherits):
    class MetaProperty(*inherits, type):  # todo: improve implementation method
        def __call__(cls, *args, **kwargs):
            __obj = super(MetaProperty, cls).__call__(*args, **kwargs)
            __dict__ = {}
            for key, val in __obj.__dict__.items():
                if key.isupper():
                    __dict__[(_name := '__magic' + key)] = val
                    setattr(cls, key, MagicProperty(lambda self, _name=_name: getattr(self, _name),
                                                    lambda self, _val, _name=_name: setattr(self, _name, _val)))
            __obj.__dict__.update(__dict__)
            return __obj

    return MetaProperty

# /__init__/Topologies/activationFunction.py
pass  # from typing import Union
pass  # from abc import ABCMeta, abstractmethod

pass  # import numpy as np


class BaseActivationFunction(metaclass=ABCMeta):
    ONE = np.float32(1)
    E = np.float32(np.e)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def activation(self, x: np.ndarray) -> "np.ndarray":
        pass

    @abstractmethod
    def activatedDerivative(self, activatedX: np.ndarray) -> "np.ndarray":
        pass


class Sigmoid(BaseActivationFunction):
    def __repr__(self):
        smooth = self.SMOOTH
        offset = self.OFFSET
        return f"{super(Sigmoid, self).__repr__()[:-1]}: {smooth=}: {offset=}>"

    def __init__(self, smooth: Union[int, float] = 1, offset: Union[int, float] = 0):
        self.SMOOTH = np.float32(smooth)
        self.OFFSET = np.float32(offset)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return self.ONE / (self.ONE + self.E ** (-self.SMOOTH * (x - self.OFFSET)))

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.SMOOTH * (activatedX * (self.ONE - activatedX))


class TanH(BaseActivationFunction):
    def __repr__(self):
        alpha = self.ALPHA
        return f"{super(TanH, self).__repr__()[:-1]}: {alpha=}>"

    def __init__(self, alpha: Union[int, float] = 1):
        self.ALPHA = np.float32(alpha)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.arctan(self.ALPHA * x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ALPHA * np.square(np.cos(activatedX))


class Relu(BaseActivationFunction):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ONE * (activatedX != 0)


class PRelu(BaseActivationFunction):
    def __repr__(self):
        leak = self.LEAK
        return f"{super(PRelu, self).__repr__()[:-1]}: {leak=}>"

    def __init__(self, leak: Union[int, float] = 0.01):
        if leak < 0: raise ValueError("parameter 'leak' cannot be less than zero")
        self.LEAK = np.float32(leak)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.LEAK * x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return np.where(activatedX <= 0, self.LEAK, self.ONE)


class Elu(BaseActivationFunction):
    def __repr__(self):
        alpha = self.ALPHA
        return f"{super(Elu, self).__repr__()[:-1]}: {alpha=}>"

    def __init__(self, alpha: Union[int, float] = 1):
        if alpha < 0: raise ValueError("parameter 'alpha' cannot be less than zero")
        self.ALPHA = np.float32(alpha)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.ALPHA * (self.E ** x - 1))

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return np.where(activatedX <= 0, activatedX + self.ALPHA, self.ONE)


class SoftMax(BaseActivationFunction):
    def activation(self, x: np.ndarray) -> np.ndarray:
        numerator = self.E ** (x - x.max(axis=-2, keepdims=True))
        return numerator / numerator.sum(axis=-2, keepdims=1)

    def activatedDerivative(self, activatedX: np.ndarray):
        jacobian = np.einsum('...ij,...kj->...jik', activatedX, activatedX, optimize='greedy')
        diagIndexes = np.diag_indices(jacobian.shape[-1])
        jacobian[..., diagIndexes[0], diagIndexes[1]] = \
            (activatedX * (1 - activatedX)).transpose().reshape(jacobian.shape[:-1])
        return jacobian.sum(axis=-1).transpose().reshape(activatedX.shape)


class SoftPlus(BaseActivationFunction):
    def activation(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.ONE + self.E ** x)

    def activatedDerivative(self, activatedX: np.ndarray) -> np.ndarray:
        return self.ONE - self.E ** -activatedX

# /__init__/Topologies/initializer.py
pass  # from abc import ABCMeta, abstractmethod
pass  # from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # from ..NeuralNetworks import *

pass  # import numpy as np


class BaseInitializer(metaclass=ABCMeta):
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.rnd = np.random.default_rng()

    def __call__(self, shape: "Base.Shape") -> "np.ndarray":
        return self._initialize(shape)

    @abstractmethod
    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        pass


class Uniform(BaseInitializer):
    def __repr__(self):
        start = self.start
        stop = self.stop
        return f"{super(Uniform, self).__repr__()[:-1]}: {start=}: {stop}>"

    def __init__(self, start: "float" = -1, stop: "float" = 1):
        super(Uniform, self).__init__()
        self.start = start
        self.stop = stop

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.uniform(self.start, self.stop, shape.HIDDEN).astype(dtype=np.float32)


class Normal(BaseInitializer):
    def __repr__(self):
        scale = self.scale
        return f"{super(Normal, self).__repr__()[:-1]}: {scale=}>"

    def __init__(self, scale: "float" = 1):
        super(Normal, self).__init__()
        self.scale = scale

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * self.scale


class Xavier(BaseInitializer):
    def __repr__(self):
        he = self.he
        return f"{super(Xavier, self).__repr__()[:-1]}: {he=}>"

    def __init__(self, he: "float" = 1):
        super(Xavier, self).__init__()
        self.he = he

    def _initialize(self, shape: "Base.Shape") -> "np.ndarray":
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * (self.he / np.prod(shape.INPUT)) ** 0.5


class NormalizedXavier(BaseInitializer):
    def __repr__(self):
        he = self.he
        return f"{super(NormalizedXavier, self).__repr__()[:-1]}: {he=}>"

    def __init__(self, he: "float" = 6):
        super(NormalizedXavier, self).__init__()
        self.he = he

    def _initialize(self, shape: "Base.Shape"):
        return self.rnd.standard_normal(shape.HIDDEN, dtype=np.float32) * (
                self.he / (np.prod(shape.INPUT) + np.prod(shape.OUTPUT)) ** (
                2 / (len(shape.INPUT) + len(shape.OUTPUT)))) ** 0.5

# /__init__/Topologies/lossFunction.py
pass  # from abc import ABCMeta, abstractmethod

pass  # import numpy as np


class BaseLossFunction(metaclass=ABCMeta):
    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __call__(self, output, target):
        return self._eval(output, target)

    @abstractmethod
    def _eval(self, output, target):
        pass


class MeanSquare(BaseLossFunction):
    def _eval(self, output, target):
        delta = output - target
        return (delta * delta).sum(axis=1).mean(), delta


class CrossEntropy(BaseLossFunction):
    def _eval(self, output, target):
        cross = -np.log(output) * target
        return (cross * cross).sum(axis=1).mean(), output - target

# /__init__/Topologies/optimizer.py
pass  # from typing import Callable
pass  # from abc import ABCMeta, abstractmethod

pass  # import numpy as np
pass  # import numexpr as ne


class BaseOptimizer(metaclass=ABCMeta):
    __args, __kwargs = (), {}
    ZERO, ONE = np.float32(0), np.float32(1)

    def __repr__(self):
        lr = self.LEARNING_RATE
        return f"<{self.__class__.__name__}:{lr=}>"

    def __new__(cls, *args, **kwargs):
        cls.__args, cls.__kwargs = args, kwargs
        obj = super(BaseOptimizer, cls).__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    @classmethod
    def __new_copy__(cls):
        return cls.__new__(cls, *cls.__args, *cls.__kwargs)

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

# /__init__/tools/helperFunction.py
pass  # import sys
pass  # import time
pass  # from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # import numpy as np

pass  # from .helperClass import PrintCols


def copyNumpyList(lis: list["np.ndarray"]):
    copyList = []
    for array in lis: copyList.append(array.copy())
    return copyList


def iterable(var):
    try:
        iter(var)
        return True
    except TypeError:
        return False


def secToHMS(seconds, hms=('h', 'm', 's')):
    encode = f'%S{hms[2]}'
    if (tim := time.gmtime(seconds)).tm_min != 0: encode = f'%M{hms[1]}' + encode
    if tim.tm_hour != 0: encode = f'%H{hms[0]}' + encode
    return time.strftime(encode, tim)


def statPrinter(key, value, *, prefix='', suffix=PrintCols.CEND, end=' '):
    print(prefix + f"{key}:{value}" + suffix, end=end)


def getSize(obj, seen=None, ref=''):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None: seen = set()
    if (obj_id := id(obj)) in seen: return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    ref += str(obj.__class__)
    if isinstance(obj, dict):
        size += sum([getSize(obj[k], seen, ref + str(k)) for k in obj.keys()])
        size += sum([getSize(k, seen, ref) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += getSize(obj.__dict__, seen, ref)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([getSize(i, seen, ref) for i in obj])

    if size > 1024 * 10:  # show files > 10Mb
        print(obj.__class__, size)
        print(ref, '\n')

    return size

# /__init__/tools/__init__.py
pass  # from .base import BaseSave, BaseLoad, Plot
pass  # from .helperClass import NumpyDataCache, Collections, PrintCols
pass  # from .helperFunction import copyNumpyList, iterable, secToHMS, statPrinter, getSize
pass  # from .magicProperty import MagicBase, MagicProperty, makeMetaMagicProperty

__all__ = [
    "BaseSave", "BaseLoad", "Plot",
    "NumpyDataCache", "Collections", "PrintCols", "copyNumpyList", "iterable", "secToHMS", "statPrinter", "getSize",
    "MagicBase", "MagicProperty", "makeMetaMagicProperty",
]

# /__init__/Topologies/dataBase.py
pass  # import warnings
pass  # from typing import Iterable, Sized, Generator

pass  # import numpy as np
pass  # import numexpr as ne

pass  # from ..tools import NumpyDataCache, BaseSave, BaseLoad, Plot


class DataBase(BaseSave, BaseLoad):
    """

    """
    DEFAULT_DIR = 'DataSets'
    DEFAULT_NAME = 'db'
    FILE_TYPE = '.zdb'
    LARGE_VAL = 5

    def __repr__(self):
        Shape = {'SIZE': self.size, 'BatchSize': self.batchSize, 'INPUT': self.inpShape, 'TARGET': self.tarShape}
        return f"<{self.__class__.__name__}:{self.NAME}:{Shape=}>"

    def __str__(self):
        HotEncode = {'INPUT': self.hotEncodeInp, 'TARGET': self.hotEncodeTar}
        SetDType = {'INPUT': self.inputSetDType, 'TARGET': self.targetSetDType}
        NormFactor = {'INPUT': f"Max:{(Max := self.inputMax)}, "
                               f"Norm:{(Norm := self.inputSetNormFactor)}, {Max*Norm=}",
                      'TARGET': f"Max:{(Max := self.targetMax)}, "
                                f"Norm:{(Norm := self.targetSetNormFactor)}, {Max*Norm=}"}
        return f"{self.__repr__()[:-1]}:\n\t{HotEncode=}\n\t{SetDType=}\n\t{NormFactor=}>"

    def saveName(self) -> str:
        return f"{self.size}s_{self.inpShape}i_{self.tarShape}o"

    def save(self, file: str = None, replace: bool = False) -> str:
        dumpFile = super(DataBase, self).save(file, replace)
        saveInputSet = self.inputs * self.inputSetNormFactor
        if self.hotEncodeInp: saveInputSet = self.oneHotDecode(saveInputSet)
        saveTargetSet = self.targets * self.targetSetNormFactor
        if self.hotEncodeTar: saveTargetSet = self.oneHotDecode(saveTargetSet)
        np.savez_compressed(dumpFile, inputSet=saveInputSet.astype(self.inputSetDType),
                            targetSet=saveTargetSet.astype(self.targetSetDType))
        return dumpFile

    @classmethod
    def load(cls, file: str, *DataBase_args, **DataBase_kwargs) -> "DataBase":
        f"""
        :param file: path like or name
        :param DataBase_args: to {DataBase.__init__}(normalizeInp, normalizeTar, reshapeInp, reshapeTar,
        oneHotMaxInp, oneHotMaxTar, name)
        """
        loadFile = super(DataBase, cls).load(file)
        nnLoader = np.load(loadFile, mmap_mode='r')
        inputSet, targetSet = nnLoader['inputSet'], nnLoader['targetSet']

        return DataBase(inputSet, targetSet, *DataBase_args, **DataBase_kwargs)

    def __getitem__(self, item):
        return self.inputs[(i := self.indices[item])], self.targets[i]

    def __init__(self,
                 inputSet: Iterable and Sized, targetSet: Iterable and Sized,
                 normalizeInp: float = None, normalizeTar: float = None,
                 reshapeInp=None, reshapeTar=None,
                 oneHotMaxInp=None, oneHotMaxTar=None,
                 name: str = ''):
        if (size := len(inputSet)) != len(targetSet): raise Exception("Both input and target set must be of same size")
        self.NAME = name
        self.inputSetDType, self.targetSetDType = inputSet.dtype, targetSet.dtype
        self.hotEncodeInp = self.hotEncodeTar = False
        if len(np.shape(inputSet)) == 1: inputSet, self.hotEncodeInp = self.oneHotEncode(inputSet, oneHotMaxInp)
        if len(np.shape(targetSet)) == 1: targetSet, self.hotEncodeTar = self.oneHotEncode(targetSet, oneHotMaxTar)
        if (maxI := np.max(inputSet)) >= self.LARGE_VAL and normalizeInp is None and not self.hotEncodeInp:
            warnings.showwarning(f"inputSet has element(s) with values till {maxI} which may cause nan training, "
                                 f"use of param 'normalizeInp=<max>' is recommended", FutureWarning, 'dataBase.py', 0)
        if (maxT := np.max(targetSet)) >= self.LARGE_VAL and normalizeTar is None and not self.hotEncodeTar:
            warnings.showwarning(f"targetSet has element(s) with values till {maxT} which may cause nan training, "
                                 f"use of param 'normalizeTar=<max>' is recommended", FutureWarning, 'dataBase.py', 0)

        inputSet, self.inputSetNormFactor = self.normalize(np.array(inputSet, dtype=np.float32), normalizeInp)
        targetSet, self.targetSetNormFactor = self.normalize(np.array(targetSet, dtype=np.float32), normalizeTar)
        self.inputMax, self.targetMax = inputSet.max(), targetSet.max()
        if reshapeInp is not None: inputSet = inputSet.reshape((size, *reshapeInp))
        if reshapeTar is not None: inputSet = targetSet.reshape((size, *reshapeTar))
        self.inputs, self.targets = NumpyDataCache(inputSet), NumpyDataCache(targetSet)

        self.size: int = size
        self.inpShape, self.tarShape = inputSet.shape[1:], targetSet.shape[1:]

        self.pointer: int = 0
        self.block: bool = False
        self.batchSize: int = 1
        self.indices = list(range(self.size))

    @staticmethod
    def oneHotEncode(_1dArray, oneHotMax=None):
        if oneHotMax is None: oneHotMax = max(_1dArray) + 1
        hotEncodedArray = np.zeros((len(_1dArray), oneHotMax, 1))
        hotEncodedArray[np.arange(hotEncodedArray.shape[0]), _1dArray] = 1

        return hotEncodedArray, oneHotMax

    @staticmethod
    def oneHotDecode(_3dArray):
        return np.where(_3dArray == 1)[1]

    # normalize input and target sets within the range of -scale to +scale
    @staticmethod
    def normalize(data, scale: float = None) -> tuple["np.ndarray", float]:
        if scale is None:
            factor = 1
        else:
            factor = ne.evaluate("abs(data) * scale", local_dict={'data': data, 'scale': scale}).max()

        return data / factor, factor

    # shuffle the index order
    def randomize(self) -> "None":
        np.random.shuffle(self.indices)

    # returns a generator for input and target sets, each batch-sets of size batchSize at a time
    # send signal '-1' to end generator
    def batchGenerator(self, batchSize) -> Generator[tuple["np.ndarray", "np.ndarray"], None, None]:
        if self.block:
            raise PermissionError("Access Denied: DataBase currently in use, "
                                  "end previous generator before creating a new one\n"
                                  "send signal '-1' to end generator or reach StopIteration")
        self.block = True
        self.batchSize = batchSize
        self.randomize()

        def generator() -> Generator:
            signal = yield
            while True:
                if signal == -1 or self.pointer + batchSize >= self.size:
                    rVal = self.__batch()
                    self.__resetVars()
                    yield rVal
                    return
                signal = yield self.__batch()
                self.pointer += batchSize

        gen = generator()
        gen.send(None)
        return gen

    def __batch(self) -> tuple[np.ndarray, np.ndarray]:
        indices = self.indices[self.pointer:self.pointer + self.batchSize]
        inputBatch = self.inputs[indices]
        targetBatch = self.targets[indices]

        return inputBatch, targetBatch

    # resets generator flags after generator cycle
    def __resetVars(self):
        self.pointer = 0
        self.block = False
        self.batchSize = None


class PlotDataBase(Plot):
    @staticmethod
    def showMap():
        pass

# /__init__/Topologies/__init__.py
pass  # from ..tools import Collections
pass  # from .dataBase import DataBase, PlotDataBase


class Activators(Collections):
    pass  # from .activationFunction import BaseActivationFunction, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus
    Base, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus = \
        BaseActivationFunction, Sigmoid, TanH, Relu, PRelu, Elu, SoftMax, SoftPlus

    def __init__(self, *activationFunctions: "Activators.Base"):
        super(Activators, self).__init__(*activationFunctions)


class Initializers(Collections):
    pass  # from .initializer import BaseInitializer, Uniform, Normal, Xavier, NormalizedXavier
    Base, Uniform, Normal, Xavier, NormalizedXavier = \
        BaseInitializer, Uniform, Normal, Xavier, NormalizedXavier

    def __init__(self, *initializer: "Initializers.Base"):
        super(Initializers, self).__init__(*initializer)


class Optimizers(Collections):
    pass  # from .optimizer import BaseOptimizer, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam
    Base, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmpProp, AdaDelta, Adam = \
        BaseOptimizer, GradientDecent, Decay, Momentum, NesterovMomentum, AdaGrad, RmsProp, AdaDelta, Adam

    def __init__(self, *optimizers: "Optimizers.Base"):
        super(Optimizers, self).__init__(*optimizers)


class LossFunction:
    pass  # from .lossFunction import BaseLossFunction, MeanSquare
    Base, MeanSquare = BaseLossFunction, MeanSquare


__all__ = [
    "Activators", "Initializers", "LossFunction", "Optimizers",
    "DataBase", "PlotDataBase"
]

# /__init__/NeuralNetworks/base.py
pass  # import time
pass  # import warnings
pass  # import cProfile
pass  # import traceback
pass  # import pstats
pass  # import os
pass  # from abc import ABCMeta, abstractmethod
pass  # from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    pass  # from ..tools import *
    pass  # from ..Topologies import *

pass  # import numpy as np

pass  # from ..tools import MagicBase, MagicProperty, makeMetaMagicProperty, PrintCols, iterable, secToHMS, statPrinter
pass  # from ..Topologies import Activators, Initializers, Optimizers, LossFunction, DataBase


class BaseShape(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.NUM_LAYERS}:{self.RAW_SHAPES}>"

    def __save__(self):
        pass

    def __getitem__(self, item):
        shapes = self.RAW_SHAPES[item]
        return self.__class__(*shapes) if isinstance(item, slice) and shapes else self.SHAPES[item]

    def __hash__(self):
        return hash(self.SHAPES)

    def __init__(self, *shapes):
        """do not change the signature of __init__"""
        self.RAW_SHAPES = shapes
        self.SHAPES = self._formatShapes(shapes)
        assert hash(self.SHAPES)
        self.NUM_LAYERS = len(self.SHAPES)
        self.INPUT = self.SHAPES[0]
        self.HIDDEN = self.SHAPES[1:-1]
        self.OUTPUT = self.SHAPES[-1]

    @staticmethod
    @abstractmethod
    def _formatShapes(shapes) -> tuple:
        """
        method to format given shapes
        :return: hashable formatted shapes
        """


class UniversalShape(BaseShape):
    """Allows any shape format, creates 'BaseShape' like object"""

    @staticmethod
    def _formatShapes(shapes) -> tuple:
        if iterable(shapes):
            assert len(shapes) > 0
            formattedShape = []
            for s in shapes:
                formattedShape.append(UniversalShape._formatShapes(s))
            return tuple(formattedShape)
        else:
            return shapes


class BaseLayer(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.SHAPE}: Ini={self.INITIALIZER}: Opt={self.optimizer}: " \
               f"AF={self.ACTIVATION_FUNCTION}>"

    def __str__(self):
        DEPS = ': '.join(f"{dName}:shape{getattr(self, dName).shape}" for dName in self.DEPS)
        return f"{self.__repr__()[:-1]}:\n{DEPS=}>"

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape",
                 initializer: "Initializers.Base",
                 optimizer: "Optimizers.Base",
                 activationFunction: "Activators.Base",
                 *depArgs, **depKwargs):
        """
        :param shape: input, output, intermediate(optional) structure of the layer
        """
        self.SHAPE = shape
        self.INITIALIZER = initializer
        self.optimizer = optimizer
        self.ACTIVATION_FUNCTION = activationFunction

        self.input = np.zeros((1, *self.SHAPE[0]), dtype=np.float32)
        self.output = np.zeros((1, *self.SHAPE[-1]), dtype=np.float32)
        self.inputDelta = np.zeros((1, *self.SHAPE[-1]), dtype=np.float32)
        self.outputDelta = np.zeros((1, *self.SHAPE[0]), dtype=np.float32)

        self.DEPS = self._defineDeps(*depArgs, **depKwargs)

    def forPass(self, _input: "np.ndarray") -> "np.ndarray":
        f"""
        method for forward pass of inputs
        :param _input: self.output from the lower layer
        :return: {self.output}
        """
        self.input = _input
        self.output = self._fire()
        return self.output

    def backProp(self, _delta: "np.ndarray") -> "np.ndarray":
        f"""
        method for back propagation of deltas
        :param _delta: value for {self.inputDelta} from {self.outputDelta} of the higher layer
        :return: {self.outputDelta}
        """
        self.inputDelta = _delta
        self.outputDelta = self._wire()
        return self.outputDelta

    def changeOptimizer(self, optimizer: "Optimizers.Base"):
        self.optimizer = optimizer
        self._initializeDepOptimizer()

    @abstractmethod
    def _initializeDepOptimizer(self):
        f"""create new optimizer instance for each dep in {self.DEPS} by using {self.optimizer.__new_copy__()}"""

    @abstractmethod
    def _defineDeps(self, *depArgs, **depKwargs) -> list['str']:
        f"""
        define all dependant objects ($DEPS) for the layer
        :return: value for {self.DEPS}
        """

    @abstractmethod
    def _fire(self) -> "np.ndarray":
        f"""
        :return: value for {self.output}, is input for the higher layer
        """

    @abstractmethod
    def _wire(self) -> "np.ndarray":
        f"""
        :return: value for {self.outputDelta}, is delta for the lower layer
        """


class BasePlot(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """


class Network:
    """

    """

    def __repr__(self):
        LossFunction = self.LOSS_FUNCTION  # noqa
        return f"{self.__class__.__name__}:{LossFunction=}"

    def __str__(self):
        layers = "\n\t\t".join(repr(layer) for layer in self.LAYERS)
        return f"{super(Network, self).__str__()}:\n\t\t{layers}"

    def __save__(self):
        pass

    def __init__(self, inputLayer: "BaseLayer", *layers: "BaseLayer", lossFunction: "LossFunction.Base"):
        assert len(layers) > 0
        self.LAYERS = inputLayer, *layers
        self.INPUT_LAYER = inputLayer
        self.HIDDEN_LAYERS = layers[:-1]
        self.OUTPUT_LAYER = layers[-1]
        self.LOSS_FUNCTION = lossFunction

    def changeOptimizer(self, _optimizer: Union["Optimizers.Base", "Optimizers"], index: int = None):
        f"""
        changes optimizer at index if given else changes all the optimizers to {_optimizer} or 
        uses given collection {Optimizers}
        """
        assert isinstance(_optimizer, (Optimizers, Optimizers.Base))
        if index is None:
            optimizers = _optimizer.get(len(self.LAYERS)) if isinstance(_optimizer, Optimizers) else \
                (_optimizer,) * len(self.LAYERS)
            for i, layer in enumerate(self.LAYERS):
                layer.changeOptimizer(optimizers[i])
        else:
            layer: "BaseLayer" = self.LAYERS[index]
            layer.changeOptimizer(_optimizer)

    def forwardPass(self, _input) -> "np.ndarray":
        f"""
        calls(and sends hierarchical I/O) the forPass method of all the layers
        :param _input: input for {self.INPUT_LAYER}
        :return: output of {self.OUTPUT_LAYER}
        """
        _output = self.INPUT_LAYER.forPass(_input)
        for layer in self.HIDDEN_LAYERS: _output = layer.forPass(_output)
        return self.OUTPUT_LAYER.forPass(_output)

    def backPropagation(self, _delta) -> "np.ndarray":
        f"""
        calls(and sends hierarchical I/O) the backProp method of all the layers
        :param _delta: delta for {self.OUTPUT_LAYER}
        :return: delta of {self.INPUT_LAYER}
        """
        _delta = self.OUTPUT_LAYER.backProp(_delta)
        for reversedLayer in self.HIDDEN_LAYERS[::-1]: _delta = reversedLayer.backProp(_delta)
        return self.INPUT_LAYER.backProp(_delta)


class BaseNN(MagicBase, metaclass=makeMetaMagicProperty(ABCMeta)):
    """

    """
    STAT_PRINT_INTERVAL = 1
    __optimizers = Optimizers(Optimizers.Adam(), ..., Optimizers.AdaGrad())

    @MagicProperty
    def optimizers(self):
        return self.__optimizers

    @optimizers.setter
    def optimizers(self, _optimizers: "Optimizers"):
        self.__optimizers = _optimizers
        self.NETWORK.changeOptimizer(self.__optimizers)

    def __repr__(self):
        Shape = self.SHAPE
        Cost, Time, Epochs = self.costTrained, secToHMS(self.timeTrained), self.epochTrained
        acc = int(self.testAccuracy), int(self.accuracyTrained)
        return f"<{self.__class__.__name__}:Acc={acc[0]}%,{acc[1]}%: {Cost=:07.4f}: {Time=}: {Epochs=}: {Shape=}>"

    def __str__(self):
        Optimizers = self.optimizers  # noqa
        TrainDataBase, TestDataBase = self.trainDataBase, self.testDataBase
        return f"{self.__repr__()[1:-1]}:\n\t{Optimizers=}\n\t{TrainDataBase=}\n\t{TestDataBase=}\n\t{self.NETWORK}"

    def __save__(self):
        pass

    def __init__(self, shape: "BaseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Base" = None):
        if initializers is None: initializers = Initializers(Initializers.Xavier(2), ..., Initializers.Xavier())
        if activators is None: activators = Activators(Activators.PRelu(), ..., Activators.SoftMax())
        if lossFunction is None: lossFunction = LossFunction.MeanSquare()

        self.SHAPE = shape

        self.costHistory, self.accuracyHistory = [], []
        self.accuracyTrained = self.testAccuracy = 0
        self.costTrained = self.timeTrained = self.epochTrained = 0

        self.numEpochs = self.batchSize = 1
        self.epoch = self.batch = 0
        self.numBatches = None
        self.training = self.profiling = False
        self.trainDataBase = self.testDataBase = None

        self.NETWORK = self._constructNetwork(initializers, activators, lossFunction)

    @abstractmethod
    def _constructNetwork(self, initializers: "Initializers" = None,
                          activators: "Activators" = None,
                          lossFunction: "LossFunction.Base" = None) -> "Network":
        pass

    def process(self, _input) -> "np.ndarray":
        if self.training:
            warnings.showwarning("processing while training in progress may have unintended conflicts",
                                 ResourceWarning, 'neuralNetwork.py->AbstractNeuralNetwork.process', 0)
            return np.NAN
        return self.NETWORK.forwardPass(np.array(_input))

    def profile(self):
        self.profiling = True
        prof = cProfile.Profile()
        prof.runctx("self._train()", locals=locals(), globals=globals())
        prof.print_stats('cumtime')
        prof.dump_stats('profile.pstat')
        with open('profile.txt', 'w') as stream:
            stats = pstats.Stats('profile.pstat', stream=stream)
            stats.sort_stats('cumtime')
            stats.print_stats()
        os.remove('profile.txt')
        self.profiling = False

    def _train(self):
        statPrinter('Epoch', f"0/{self.numEpochs}", prefix=PrintCols.CBOLDITALICURL + PrintCols.CBLUE)
        self.training = True
        if self.epochTrained == 0:
            loss, _, acc = self._trainer(self.trainDataBase[:self.batchSize])
            trainCosts, trainAccuracies = [loss], [acc]
        else:
            trainCosts, trainAccuracies = [self.costHistory[-1][-1]], [self.accuracyHistory[-1][-1]]

        for self.epoch in range(1, self.numEpochs + 1):
            epochTime = nextPrintTime = 0
            costTrained = accuracyTrained = 0
            try:
                for self.batch, _batch in enumerate(self.trainDataBase.batchGenerator(self.batchSize)):
                    timeStart = time.time()
                    loss, delta, acc = self._trainer(_batch)
                    self.NETWORK.backPropagation(delta)
                    costTrained += loss
                    accuracyTrained += acc
                    batchTime = time.time() - timeStart
                    epochTime += batchTime
                    if epochTime >= nextPrintTime or self.batch == self.numBatches - 1:
                        nextPrintTime += self.STAT_PRINT_INTERVAL
                        self.printStats(costTrained / (self.batch + 1), trainCosts[-1],
                                        accuracyTrained / (self.batch + 1), trainAccuracies[-1], epochTime)
                self.timeTrained += epochTime
                self.epochTrained += 1
                self.costTrained = costTrained / self.numBatches
                self.accuracyTrained = accuracyTrained / self.numBatches
                trainCosts.append(self.costTrained)
                trainAccuracies.append(self.accuracyTrained)
            except Exception:  # noqa
                traceback.print_exc()
                warnings.showwarning("unhandled exception occurred while training,"
                                     "\nquiting training and rolling back to previous auto save", RuntimeWarning,
                                     'base.py', 0)
                raise NotImplementedError  # todo: roll back and auto save
        self.costHistory.append(trainCosts)
        self.accuracyHistory.append(trainAccuracies)
        self.training = False
        statPrinter('', '', end='\n')

    def train(self, epochs: int = None,
              batchSize: int = None,
              trainDataBase: "DataBase" = None,
              optimizers: "Optimizers" = None,
              profile: bool = False,
              test: Union[bool, "DataBase"] = None):
        # todo: implement "runs"
        if epochs is not None: self.numEpochs = epochs
        if batchSize is not None: self.batchSize = batchSize
        if trainDataBase is not None: self.trainDataBase = trainDataBase
        if optimizers is not None: self.optimizers = optimizers
        assert isinstance(self.trainDataBase, DataBase)
        assert self.trainDataBase.inpShape == self.SHAPE.INPUT and self.trainDataBase.tarShape == self.SHAPE.OUTPUT
        if trainDataBase is not None or batchSize is not None:
            self.numBatches = int(np.ceil(self.trainDataBase.size / self.batchSize))
        if profile:
            self.profile()
        else:
            self._train()
        if test or test is None: self.test(test)

    def printStats(self, loss, prevLoss, acc, prevAcc, epochTime):
        print(end='\r')
        """__________________________________________________________________________________________________________"""
        statPrinter('Epoch', f"{self.epoch:0{len(str(self.numEpochs))}d}/{self.numEpochs}",
                    prefix=PrintCols.CBOLDITALICURL + PrintCols.CBLUE, suffix='')
        statPrinter('Batch', f"{(b := self.batch + 1):0{len(str(self.numBatches))}d}/{self.numBatches}",
                    suffix='', end='')
        statPrinter(f"({int(b / self.numBatches * 100):03d}%)", '')
        """__________________________________________________________________________________________________________"""
        statPrinter('Cost', f"{loss:07.4f}", prefix=PrintCols.CYELLOW, suffix='')
        statPrinter('Cost-Dec', f"{(prevLoss - loss):07.4f}", suffix='')
        statPrinter('Acc', f"{int(acc):03d}%", prefix=PrintCols.CYELLOW, suffix='')
        statPrinter('Acc-Inc', f"{int(acc - prevAcc):03d}%")
        """__________________________________________________________________________________________________________"""
        elapsed = self.timeTrained + epochTime
        avgTime = elapsed / (effectiveEpoch := self.epoch - 1 + (self.batch + 1) / self.numBatches)
        statPrinter('Time', secToHMS(elapsed), prefix=PrintCols.CBOLD + PrintCols.CRED2, suffix='')
        statPrinter('Epoch-Time', secToHMS(epochTime), suffix='')
        statPrinter('Avg-Time', secToHMS(avgTime), suffix='')
        statPrinter('Eta', secToHMS(avgTime * (self.numEpochs - effectiveEpoch)))
        """__________________________________________________________________________________________________________"""

    def _trainer(self, _batch: tuple["np.ndarray", "np.ndarray"]):
        output, target = self.NETWORK.forwardPass(_batch[0]), _batch[1]
        loss, delta = self.NETWORK.LOSS_FUNCTION(output, target)
        acc = self._tester(output, target)
        return loss, delta, acc

    def accuracy(self, inputSet, targetSet):
        assert (size := np.shape(inputSet)[0]) == np.shape(targetSet)[0], \
            "the size of both inputSet and targetSet should be same"
        try:
            return self._tester(self.process(inputSet), targetSet)
        except MemoryError:
            accuracy1 = self.accuracy(inputSet[:(to := size // 2)], targetSet[:to])
            accuracy2 = self.accuracy(inputSet[to:], targetSet[to:])
            return (accuracy1 + accuracy2) / 2

    def test(self, testDataBase: "DataBase" = None):
        statPrinter('Testing', 'wait...', prefix=PrintCols.CBOLD + PrintCols.CYELLOW, suffix='')
        if self.trainDataBase is not None:
            self.accuracyTrained = self.accuracy(self.trainDataBase.inputs, self.trainDataBase.targets)
        if testDataBase is not None:
            assert isinstance(testDataBase, DataBase)
            assert testDataBase.inpShape == self.SHAPE.INPUT and testDataBase.tarShape == self.SHAPE.OUTPUT
            self.testAccuracy = self.accuracy(testDataBase.inputs, testDataBase.targets)
        print(end='\r')
        statPrinter('Train-Accuracy', f"{self.accuracyTrained}%", suffix='', end='\n')
        statPrinter('Test-Accuracy', f"{self.testAccuracy}%", end='\n')

    @staticmethod
    def _tester(_output: "np.ndarray", _target: "np.ndarray") -> "np.ndarray":
        if np.shape(_target) != 1:
            # poly node multi classification
            outIndex = np.argmax(_output, axis=1)
            targetIndex = np.argmax(_target, axis=1)
        else:
            # single node binary classification
            outIndex = _output.round()
            targetIndex = _target
        result: "np.ndarray" = outIndex == targetIndex
        return result.mean() * 100

# /__init__/NeuralNetworks/dense.py
pass  # from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # from ..Topologies import *

pass  # import numpy as np

pass  # from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape, Network


class DenseShape(BaseShape):
    """

    """

    def __init__(self, *shapes: int):
        super(DenseShape, self).__init__(*shapes)

    @staticmethod
    def _formatShapes(shapes) -> tuple:
        assert len(shapes) > 0
        formattedShape = []
        for s in shapes:
            assert isinstance(s, int) and s > 0, "all args of *shapes must be integers > 0"
            formattedShape.append((s, 1))
        return tuple(formattedShape)


class DenseLayer(BaseLayer):  # todo: pre-set deltas after forwardPass
    """

    """

    def __save__(self):
        return super(DenseLayer, self).__save__()

    def _initializeDepOptimizer(self):
        self.weightOptimizer = self.optimizer.__new_copy__()
        self.biasesOptimizer = self.optimizer.__new_copy__()

    def _defineDeps(self) -> list['str']:
        self.weights = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], self.SHAPE.INPUT[0]),
                                                       self.SHAPE.OUTPUT))
        self.biases = self.INITIALIZER(UniversalShape(self.SHAPE.INPUT, *(self.SHAPE.OUTPUT[0], 1), self.SHAPE.OUTPUT))
        self.delta = None
        self.activeDerivedDelta = None
        self._initializeDepOptimizer()
        return ['weights', 'biases']

    def __gradWeights(self, weights):  # BottleNeck
        self.delta = np.einsum('oi,...oj->...ij', weights, self.inputDelta, optimize='greedy')
        self.activeDerivedDelta = \
            np.einsum('...ij,...ij->...ij', self.inputDelta, self.ACTIVATION_FUNCTION.activatedDerivative(self.output),
                      optimize='greedy')
        return np.einsum('...ij,...oj->oi', self.input, self.activeDerivedDelta, optimize='greedy')

    def __gradBiases(self, _=None):
        return self.activeDerivedDelta.sum(axis=0)

    def _fire(self) -> "np.ndarray":  # BottleNeck
        return self.ACTIVATION_FUNCTION.activation(
            np.einsum('oi,...ij->...oj', self.weights, self.input, optimize='greedy') + self.biases)

    def _wire(self) -> "np.ndarray":
        self.weights -= self.weightOptimizer(self.__gradWeights, self.weights)
        self.biases -= self.biasesOptimizer(self.__gradBiases, self.biases)
        return self.delta


class DensePlot(BasePlot):
    """

    """


class DenseNN(BaseNN):
    """

    """

    def __str__(self):
        return super(DenseNN, self).__str__()

    def __save__(self):
        return super(DenseNN, self).__save__()

    def __init__(self, shape: "DenseShape",
                 initializers: "Initializers" = None,
                 activators: "Activators" = None,
                 lossFunction: "LossFunction.Base" = None):
        super(DenseNN, self).__init__(shape, initializers, activators, lossFunction)

    def _constructNetwork(self, initializers: "Initializers" = None,
                          activators: "Activators" = None,
                          lossFunction: "LossFunction.Base" = None) -> "Network":
        layers = []
        for i, _initializer, _optimizer, _aF in zip(range(_length := self.SHAPE.NUM_LAYERS - 1),
                                                    initializers(_length),
                                                    self.optimizers(_length),  # noqa
                                                    activators(_length)):
            layers.append(DenseLayer(self.SHAPE[i:i + 2], _initializer, _optimizer, _aF))
        return Network(*layers, lossFunction=lossFunction)

# /__init__/NeuralNetworks/conv.py
pass  # from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # from ..tools import *
    pass  # from ..Topologies import *

pass  # import numpy as np

pass  # from .base import BaseShape, BaseLayer, BasePlot, BaseNN, UniversalShape, Network


class ConvShape:
    pass


class ConvLayer:
    pass


class ConvPlot(BasePlot):
    """

    """


class ConvNN:
    pass

# /__init__/NeuralNetworks/__init__.py
class Base:
    pass  # from .base import BaseShape, BaseShape, BasePlot, BaseNN, UniversalShape, Network
    Shape, Layer, Plot, NN = BaseShape, BaseShape, BasePlot, BaseNN
    UniversalShape, Network = UniversalShape, Network


class Dense:
    pass  # from .dense import DenseShape, DenseLayer, DensePlot, DenseNN
    Shape, Layer, Plot, NN = DenseShape, DenseLayer, DensePlot, DenseNN


class Conv:
    pass  # from .conv import ConvShape, ConvLayer, ConvPlot, ConvNN
    Shape, Layer, Plot, NN = ConvShape, ConvLayer, ConvPlot, ConvNN


__all__ = [
    "Base", "Dense", "Conv"
]

# /__init__/__init__.py
pass  # from .NeuralNetworks import *
pass  # from .tools import *
pass  # from .Topologies import *

# /main.py
# todo: dynamic optimizers
# todo: DataBase shaping using self.SHAPE
# todo: auto hyperparameter tuning: Grid search, Population-based natural selection
# todo: auto train stop, inf train
pass  # from __init__ import *
pass  # from DataSets import dataSet
pass  # from Models import model

db = DataBase.load(TrainSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1),
                   name='TrainSets.EmnistBalanced')
db2 = DataBase.load(TestSets.EmnistBalanced, normalizeInp=1, reshapeInp=(-1, 1),
                    name='TestSets.EmnistBalanced')
# db2 = False
dense_nn = Dense.NN(shape=Dense.Shape(db.inpShape[0], *(392, 196), db.tarShape[0]),
                    initializers=None,
                    activators=None,
                    lossFunction=None)
dense_nn.train(epochs=5,
               batchSize=128,
               trainDataBase=db,
               optimizers=None,
               profile=False,
               test=db2)
print(db, db2, dense_nn, sep='\n\n')
