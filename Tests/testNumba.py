import numba as nb
import numpy as np


@nb.njit(fastmath=True, parallel=True, cache=True)
def foo(A, B):
    assert len(A.shape) == 6
    assert len(B.shape) == 4

    res = np.empty((A.shape[0], B.shape[0], *A.shape[2:4]), dtype=A.dtype)

    for m in nb.prange(A.shape[0]):
        for x in nb.prange(A.shape[2]):
            for y in nb.prange(A.shape[3]):
                for k in nb.prange(B.shape[0]):
                    acc = 0
                    for l in nb.prange(B.shape[1]):
                        for i in nb.prange(B.shape[2]):
                            for j in nb.prange(B.shape[3]):
                                acc += A[m, l, x, y, i, j] * B[k, l, i, j]
                    res[m, k, x, y] = acc

    return res


@nb.njit(fastmath=True, parallel=True, cache=True)
def foo2(A):
    assert len(A.shape) == 6

    res = np.empty(A.shape[:-2], dtype=A.dtype)

    for m in nb.prange(A.shape[0]):
        for l in nb.prange(A.shape[1]):
            for x in nb.prange(A.shape[2]):
                for y in nb.prange(A.shape[3]):
                    acc = A[m, l, x, y, 0, 0]
                    for i in nb.prange(A.shape[4]):
                        for j in nb.prange(A.shape[5]):
                            if A[m, l, x, y, i, j] > acc:
                                acc = A[m, l, x, y, i, j]
                    res[m, l, x, y] = acc

    return res
