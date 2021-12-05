import typing as tp
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import tempfile as tf
from numpy.lib import format as fm
from matplotlib import pyplot as plt, widgets as wg

if tp.TYPE_CHECKING:
    from Topologies.activationFuntion import AbstractActivationFunction


# setup for plotting
plt.style.use('dark_background')


class WBShape:
    def __init__(self, *wbShape):
        self._shape = tuple(wbShape)
        self.LAYERS = len(self._shape)

    def __getitem__(self, item):
        return self._shape[item]

    @property
    def shape(self):
        return self._shape


class Activators:
    def __init__(self, *activationFunctions: "AbstractActivationFunction"):
        self.activationFunctions = activationFunctions

    def __call__(self, length):
        return self.get(length)

    def get(self, length):
        activations = [None]
        activationDerivatives = [None]
        prevActivationFunction = None
        numEllipsis = self.activationFunctions.count(Ellipsis)
        numActivations = len(self.activationFunctions) - numEllipsis
        vacancy = length - numActivations
        for activationFunction in self.activationFunctions:
            if activationFunction == Ellipsis:
                for i in range(filled := (vacancy // numEllipsis)):
                    activations.append(prevActivationFunction.activation)
                    activationDerivatives.append(prevActivationFunction.activatedDerivative)
                vacancy -= filled
                numEllipsis -= 1
                continue
            prevActivationFunction = activationFunction
            activations.append(activationFunction.activation)
            activationDerivatives.append(activationFunction.activatedDerivative)

        return activations, activationDerivatives


class NumpyDataCache(np.ndarray):
    def __new__(cls, array):
        return cls.writeNpyCache(array)

    @staticmethod
    def writeNpyCache(array: "np.ndarray") -> np.ndarray:
        with tf.NamedTemporaryFile(suffix='.npy') as file:
            np.save(file, array)
            file.seek(0)
            fm.read_magic(file)
            fm.read_array_header_1_0(file)
            memMap = np.memmap(file, mode='r', shape=array.shape, dtype=array.dtype, offset=file.tell())

        return memMap


class AbstractSave(metaclass=ABCMeta):
    DEFAULT_DIR: str
    DEFAULT_NAME: str
    FILE_TYPE: str

    @abstractmethod
    def saveName(self) -> str:
        pass

    @abstractmethod
    def _write(self, dumpFile, *args, **kwargs):
        pass

    def save(self, file: str = None, replace: bool = False, *args, **kwargs) -> str:
        if file is None:
            file = self.DEFAULT_NAME
        if not (fpath := os.path.dirname(file)):
            fpath = os.getcwd() + self.DEFAULT_DIR
            fname = file
        else:
            fpath += '\\'
            fname = os.path.basename(file)
        os.makedirs(fpath, exist_ok=True)
        if len(fname) >= 1 + len(self.FILE_TYPE) and fname[-4:] == self.FILE_TYPE:
            fname = fname[:-4]
        savePath = fpath + fname + self.saveName()

        i = 0
        numSavePath = savePath
        if not replace:
            while 1:
                if i != 0:
                    numSavePath = savePath + ' (' + str(i) + ')'
                if os.path.exists(numSavePath + self.FILE_TYPE):
                    i += 1
                else:
                    break

        with open(finalPath := (numSavePath + self.FILE_TYPE), 'wb') as dumpFile:
            self._write(dumpFile, *args, **kwargs)

        return finalPath


class AbstractLoad(metaclass=ABCMeta):
    DEFAULT_DIR: str
    FILE_TYPE: str

    @classmethod
    @abstractmethod
    def _read(cls, loadFile, *args, **kwargs):
        pass

    @classmethod
    def load(cls, file: str, *args, **kwargs):
        if file:
            if not (fpath := os.path.dirname(file)):
                fpath = os.getcwd() + cls.DEFAULT_DIR
                fName = file
            else:
                fpath += '\\'
                fName = os.path.basename(file)
        else:
            raise NameError("file not given")
        if '.' not in fName:
            fName += cls.FILE_TYPE

        with open(fpath + fName, 'rb') as loadFile:
            rVal = cls._read(loadFile, *args, **kwargs)

        return rVal


# todo: improve Plot
class Plot:
    @staticmethod
    def plotHeight(x, y=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax, fig = ax, ax.figure
        args = (x, y)
        if y is None:
            args = (x,)
        for i in range(len(x)):
            ax.plot(*(arg[i] for arg in args), c=np.random.rand(3))

        return fig, ax

    @ staticmethod
    def plotMap(_2dVect, text='', textPos=(.01, .01), col='yellow', ax=None):
        if not isinstance(_2dVect, np.ndarray):
            _2dVect = np.array(_2dVect)
        if len(_2dVect.shape) != 2:
            raise ValueError("param '_2dVect' must have only 2Dimensions")
        if ax is None:
            fig, ax = plt.subplots()
        else:
            ax, fig = ax, ax.figure
        ax.imshow(_2dVect)
        ax.text(*textPos, text, transform=ax.transAxes, c=col)

    @staticmethod
    def plotMultiMap(_3dVect: "np.ndarray", text=None, rows=4, columns=4, size=6):
        shape = _3dVect.shape[-2:]
        MAX = rows * columns
        fig = plt.figure(figsize=(shape[0] / (figShapeFactor := max(shape)) * size, shape[1] / figShapeFactor * size))
        axes = [fig.add_subplot(rows + 1, columns, i + 1) for i in range(MAX)]
        butPrev = wg.Button(ax := fig.add_subplot(rows + 1, 2, 2 * rows + 1), '<-', color='red', hovercolor='blue')
        ax.set_aspect(.1)
        butNext = wg.Button(ax := fig.add_subplot(rows + 1, 2, 2 * rows + 2), '->', color='red', hovercolor='blue')
        ax.set_aspect(.1)
        fig.page = 0
        if text is None:
            text = range(_3dVect.shape[0])

        def onclick(_page):
            if _page == 0:
                butPrev.active = False
                butPrev.ax.patch.set_visible(False)
            else:
                butPrev.active = True
                butPrev.ax.patch.set_visible(True)
            if _page == (_3dVect.shape[0] - 1) // MAX:
                butNext.active = False
                butNext.ax.patch.set_visible(False)
            else:
                butNext.active = True
                butNext.ax.patch.set_visible(True)
            fig.page = _page
            if (to := (_page + 1) * MAX) > _3dVect.shape[0]:
                to = _3dVect.shape[0]
            [(ax.clear(), ax.set_xticks([]), ax.set_yticks([])) for ax in axes]
            for i, im in enumerate(_3dVect[_page * MAX:to]):
                Plot.plotMap(im, str(text[i]), ax=axes[i])
            fig.subplots_adjust(.01, .01, .99, .99, 0, 0)
            fig.canvas.draw()

        butNext.on_clicked(lambda *_: onclick(fig.page + 1))
        butPrev.on_clicked(lambda *_: onclick(fig.page - 1))
        onclick(fig.page)

    @staticmethod
    def show():
        plt.show()


def copyNumpyList(lis: tp.List[np.ndarray]):
    copyList = []
    for array in lis:
        copyList.append(array.copy())

    return copyList


def iterable(var):
    try:
        iter(var)
        return True
    except TypeError:
        return False
