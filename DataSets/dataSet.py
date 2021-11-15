import os as _os

import numpy as _np


class TrainSets:
    Xor = [[[0], [0], [0]], [[0], [0], [1]], [[0], [1], [0]], [[0], [1], [1]],
           [[1], [0], [0]], [[1], [0], [1]], [[1], [1], [0]], [[1], [1], [1]]], \
          [[[0], [1]], [[1], [0]], [[1], [0]], [[0], [1]], [[1], [0]], [[0], [1]], [[0], [1]], [[1], [0]]]

    _fname = "dbs112800i784o47.nndb.npz"
    _fpath = _os.getcwd() + '\\DataSets\\' + _fname
    npLoader = _np.load(_fpath)
    EmnistBalanced = [npLoader['arr_0'], npLoader['arr_1']]


class TestSets:
    Xor = TrainSets.Xor

    _fname = "dbs18800i784o47.nndb.npz"
    _fpath = _os.getcwd() + '\\DataSets\\' + _fname
    npLoader = _np.load(_fpath)
    EmnistBalanced = [npLoader['arr_0'], npLoader['arr_1']]
