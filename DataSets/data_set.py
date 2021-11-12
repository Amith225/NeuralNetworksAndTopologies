import os as _os

import numpy as _np


class TRAIN_SETS:
    train_set_xor = [[[0], [0], [0]], [[0], [0], [1]], [[0], [1], [0]], [[0], [1], [1]],
                     [[1], [0], [0]], [[1], [0], [1]], [[1], [1], [0]], [[1], [1], [1]]], \
                    [[[0]], [[1]], [[1]], [[0]], [[1]], [[0]], [[0]], [[1]]]

    _fname = "dbs112800i784o47.nndb.npz"
    _fpath = _os.getcwd() + '\\DataSets\\' + _fname
    _np_loader = _np.load(_fpath)
    train_set_emnist_balanced = [_np_loader['arr_0'], _np_loader['arr_1']]


class TEST_SETS:
    test_set_xor = [[[0], [0], [0]], [[0], [0], [1]], [[0], [1], [0]], [[0], [1], [1]],
                    [[1], [0], [0]], [[1], [0], [1]], [[1], [1], [0]], [[1], [1], [1]]], \
                   [[[0]], [[1]], [[1]], [[0]], [[1]], [[0]], [[0]], [[1]]]

    _fname = "dbs18800i784o47.nndb.npz"
    _fpath = _os.getcwd() + '\\DataSets\\' + _fname
    _np_loader = _np.load(_fpath)
    test_set_emnist_balanced = [_np_loader['arr_0'], _np_loader['arr_1']]
