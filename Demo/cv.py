import numpy as np
import cv2
import sys


def bordering(_j, _i, connect=8):
    assert connect in (4, 8)
    mi, mj, pi, pj = _i - 1, _j - 1, _i + 1, _j + 1
    if connect == 8:
        rVal = [(_j, pi), (pj, pi), (pj, _i), (pj, mi), (_j, mi), (mj, mi), (mj, _i), (mj, pi)]
    else:
        rVal = [(_j, pi), (pj, _i), (_j, mi), (mj, _i)]
    rValNew = []
    for r in rVal:
        if not (r[0] < 0 or r[1] < 0):
            rValNew.append(r)
    return tuple(rValNew)


def floodFill(src, tarCol, col, wallCol):
    y = np.where(src == wallCol)[0][-1]
    startX = 0
    while True:
        y -= 1
        x = np.where(src[y] == wallCol)[0]
        if x.shape[0] == 0: break
        nearest = np.where(x - startX > 0)[0]
        x = (x[nearest[0]] + x[nearest[0] - 1]) // 2 if nearest.shape[0] != 0 else int(x.mean())
        startX = x
        startY = y
        while True:
            x += 1
            if x < 0 or x >= src.shape[1] or y < 0 or y >= src.shape[0]: break
            if src[y][x] == wallCol or src[y][x] == col: break
            if src[y][x] != tarCol: break
            src[y][x] = col
        x = startX
        y = startY
        while True:
            if x < 0 or x >= src.shape[1] or y < 0 or y >= src.shape[0]: break
            if src[y][x] == wallCol or src[y][x] == col: break
            if src[y][x] != tarCol: break
            src[y][x] = col
            x -= 1
        x = startX
        y = startY + 1
        while True:
            if x < 0 or x >= src.shape[1] or y < 0 or y >= src.shape[0]: break
            if src[y][x] == wallCol or src[y][x] == col: break
            if src[y][x] != tarCol: break
            src[y][x] = col
            x += 1
            y += 1
            startY += 1
        x = startX
        y = startY


def drawContour(src, _contour, col=0, lineWidth=1):
    if lineWidth >= 0:
        srcCopy = src.copy()
        for line in range(lineWidth):
            Ncontour = _contour - line
            Pcontour = _contour + line
            srcCopy[Ncontour[:, 1], _contour[:, 0]] = col
            srcCopy[_contour[:, 1], Ncontour[:, 0]] = col
            srcCopy[Pcontour[:, 1], _contour[:, 0]] = col
            srcCopy[_contour[:, 1], Pcontour[:, 0]] = col
        return srcCopy
    elif lineWidth == -1:
        srcCopy = drawContour(np.zeros_like(src), _contour, 255)
        floodFill(srcCopy, 0, 255, 255)
        mask = np.where(srcCopy == 0, False, True)
        return mask
    else:
        raise ValueError("lineWidth must be >= -1")


def findContour(src, _hierarchy=None, _contours=None, offset=None, edgePix=None):
    if edgePix is None: edgePix = 0
    if _hierarchy is None: _hierarchy = [[-1, -1, -1, -1]]
    if _contours is None: _contours = []  # todo
    notEdgePIx = 255 * (not edgePix)
    edge = src == edgePix
    if np.alltrue(~edge): return _hierarchy, _contours
    start = divmod(np.argmax(edge), src.shape[0])
    contours = [[start[::-1]]]
    borders = bordering(*start)
    prev = start
    index = 0
    current = borders[index]
    while current != start and index < 8:
        if src[current] == edgePix:
            contours[-1].append(current[::-1])
            borders = bordering(*current)
            index = borders.index(prev)
            prev = current
            borders = borders[index:] + borders[:index]
            index = 0
        index += 1
        current = borders[index]
    contours = np.array(contours)
    mask = drawContour(src, contours[-1], lineWidth=-1)
    nextSrc, childSrc = src * ~mask + np.uint8(notEdgePIx) * mask, src * mask + np.uint8(edgePix) * ~mask
    print('next', edgePix)
    cv2.imshow('', nextSrc), cv2.waitKey(0), cv2.destroyAllWindows(), cv2.waitKey(1)
    print('child', edgePix)
    cv2.imshow('', childSrc), cv2.waitKey(0), cv2.destroyAllWindows(), cv2.waitKey(1)
    if offset is None:
        offset = 0
        h = [-1, -1, -1, offset]
    else:
        h = [-1, offset, -1, _hierarchy[offset][-1]]
        _hierarchy[offset][0] = offset + 1
    child = findContour(childSrc, [h], edgePix=notEdgePIx)
    sibling = findContour(nextSrc, [*_hierarchy, *child[0]], contours, offset=offset + len(child[0]) - 1,
                          edgePix=edgePix)
    print(sibling[0])
    return sibling[0], contours
