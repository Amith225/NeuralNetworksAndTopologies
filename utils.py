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


class NumpyDataCache:
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
                fname = file
            else:
                fpath += '\\'
                fname = os.path.basename(file)
        else:
            raise NameError("file not given")
        if '.' not in fname:
            fname += cls.FILE_TYPE

        with open(fpath + fname, 'rb') as loadFile:
            rVal = cls._read(loadFile, *args, **kwargs)

        return rVal


class Plot:
    @staticmethod
    def plot(x, y=None):
        fig = plt.figure()
        ax = fig.add_subplot()
        args = (x, y)
        if y is None:
            args = (x,)
        for i in range(len(x)):
            ax.plot(*(arg[i] for arg in args), c=np.random.rand(3))

        return fig, ax

    @staticmethod
    def plotInputVecAsImg(inpVec: "np.ndarray"):
        shape = inpVec.shape[1:]
        rows, columns = 4, 4
        MAX = rows * columns
        figShapeFactor = max(shape)
        fig = plt.figure(figsize=(shape[0] / figShapeFactor * 6, shape[1] / figShapeFactor * 6))
        axes = [fig.add_subplot(rows, columns, i + 1) for i in range(MAX)]
        butPrev = wg.Button(plt.axes([0, 0, .5, .05]), '<-', color='red', hovercolor='blue')
        butNext = wg.Button(plt.axes([.5, 0, .5, .05]), '->', color='red', hovercolor='blue')
        fig.page = 0

        def onclick(_page):
            if _page == 0:
                butPrev.active = False
                butPrev.ax.patch.set_visible(False)
            else:
                butPrev.active = True
                butPrev.ax.patch.set_visible(True)
            if _page == (inpVec.shape[0] - 1) // MAX:
                butNext.active = False
                butNext.ax.patch.set_visible(False)
            else:
                butNext.active = True
                butNext.ax.patch.set_visible(True)
            fig.page = _page
            to = (_page + 1) * MAX
            if to > inpVec.shape[0]:
                to = inpVec.shape[0]
            [(ax.clear(),) for ax in axes]
            for i, im in enumerate(inpVec[_page * MAX:to]):
                axes[i].spines['bottom'].set_color('.5')
                axes[i].spines['top'].set_color('.5')
                axes[i].spines['right'].set_color('.5')
                axes[i].spines['left'].set_color('.5')
                axes[i].imshow(im)
                axes[i].tick_params(axis='x', colors='.15', which='both')
                axes[i].tick_params(axis='y', colors='.15', which='both')
                axes[i].yaxis.label.set_color('.15')
                axes[i].xaxis.label.set_color('.15')
                fig.patch.set_facecolor('.15')
                axes[i].annotate(i + MAX * _page, (1, 5))
            fig.subplots_adjust(.05, .1, .95, .95)
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
