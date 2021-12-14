import numpy as np
import tempfile as tf
from numpy.lib import format as fm
from matplotlib import pyplot as plt, widgets as wg

import typing as tp
import os
import sys
from abc import ABCMeta, abstractmethod

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
            fName = file
        else:
            fpath += '\\'
            fName = os.path.basename(file)
        os.makedirs(fpath, exist_ok=True)
        if len(fName) >= 1 + len(self.FILE_TYPE) and fName[-4:] == self.FILE_TYPE:
            fName = fName[:-4]
        savePath = fpath + fName.replace(' ', '.') + '.' + self.saveName().replace(' ', '')

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


class Plot:
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
            [(ax.clear(), ax.set_xticks([]), ax.set_yticks([])) for ax in axes]
            plotter(_page, axes)
            fig.subplots_adjust(.01, .01, .99, .99, 0, 0)
            fig.canvas.draw()

        butNext.on_clicked(lambda *_: onclick(fig.page + 1))
        butPrev.on_clicked(lambda *_: onclick(fig.page - 1))
        onclick(fig.page)

        return axes

    # fixme: there are a lot of params.
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


def getSize(obj, seen=None, ref=''):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
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
