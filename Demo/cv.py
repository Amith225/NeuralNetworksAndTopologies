import numpy as np
import cv2


def getGrp(index, _contour):
    same_y_indexes = np.where(_contour[:, 1] == _contour[index][1])[0]
    e = np.sort(_contour[same_y_indexes], axis=0)
    grp = np.split(e, np.where(np.diff(e[:, 0]) != 1)[0] + 1)
    grp = np.array([[g[0], g[-1]] for g in grp])
    return same_y_indexes, grp


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
        mask = src != src
        index, maxIndex = -1, _contour.shape[0]
        indexFlags = []
        prevGrp = None
        while (index := index + 1) < maxIndex:
            if index in indexFlags: continue
            same_y_indexes, grp = getGrp(index, _contour)
            indexFlags.extend(same_y_indexes)
            if prevGrp is not None and (length := len(prevGrp) - len(grp)) > 0:
                prevGrp = grp
                grp_ = []
                for i in range(length): grp_.extend([grp[i], grp[i]])
                grp = np.array(grp_ + list(grp[length:]))
            else:
                prevGrp = grp
            if len(grp) % 2 != 0: grp = np.concatenate((grp, [grp[len(grp) - 2]]))
            for i, g in enumerate(grp[::2]):
                a, b = g[0][0], grp[i * 2 + 1][-1][0] + 1
                if a > b: a, b = b - 1, a + 1
                mask[g[0][1], a:b] = True
        return mask
    else:
        raise ValueError("lineWidth must be >= -1")


def bordering(_j, _i):
    mi, mj, pi, pj = _i - 1, _j - 1, _i + 1, _j + 1
    return [(_j, pi), (pj, pi), (pj, _i),
            (pj, mi), (_j, mi),
            (mj, mi), (mj, _i), (mj, pi)]


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
    sibling = findContour(nextSrc, [*_hierarchy, *child[0]], contours, offset=offset + len(child[0]) - 1, edgePix=edgePix)
    print(sibling[0])
    return sibling[0], contours
