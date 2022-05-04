import numpy as np
import cv2


def drawContour(src, _contour, col, lineWidth=1):
    srcCopy = src.copy()
    if lineWidth >= 0:
        for line in range(lineWidth):
            Ncontour = _contour - line
            Pcontour = _contour + line
            srcCopy[Ncontour[:, 1], _contour[:, 0]] = col
            srcCopy[_contour[:, 1], Ncontour[:, 0]] = col
            srcCopy[Pcontour[:, 1], _contour[:, 0]] = col
            srcCopy[_contour[:, 1], Pcontour[:, 0]] = col
        return srcCopy
    elif lineWidth in (-1, -2):
        if lineWidth == -2: srcCopy = np.ones_like(src) * np.uint8(col)
        index, maxIndex = -1, _contour.shape[0]
        indexFlags = []
        while (index := index + 1) < maxIndex:
            if index in indexFlags: continue
            same_y_indexes = np.where(_contour[:, 1] == _contour[index][1])[0]
            indexFlags.extend(same_y_indexes)
            e = _contour[same_y_indexes]
            e1, e2 = e.min(axis=0), e.max(axis=0)
            if lineWidth == -1:
                srcCopy[e1[1], e1[0]:e2[0] + 1] = col
            else:
                srcCopy[e1[1], e1[0]:e2[0] + 1] = src[e1[1], e1[0]:e2[0] + 1]
        return srcCopy
    else:
        raise ValueError("lineWidth must be >= -2")


def bordering(_j, _i):
    mi, mj, pi, pj = _i - 1, _j - 1, _i + 1, _j + 1
    return [(_j, pi), (pj, pi), (pj, _i),
            (pj, mi), (_j, mi),
            (mj, mi), (mj, _i), (mj, pi)]


def findContour(src, _hierarchy=None, _contours=None, offset=None, edgePix=None):
    if edgePix is None: edgePix = 0
    if _hierarchy is None: _hierarchy = [[-1, -1, -1, -1]]
    if _contours is None: _contours = []  # todo
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
    nextSrc = drawContour(src, contours[-1], 255 * (not edgePix), -1)
    childSrc = drawContour(src, contours[-1], 255 * edgePix, -2)
    print('next')
    cv2.imshow('', nextSrc), cv2.waitKey(0), cv2.destroyAllWindows(), cv2.waitKey(1)
    print('child', edgePix)
    cv2.imshow('', childSrc), cv2.waitKey(0), cv2.destroyAllWindows(), cv2.waitKey(1)
    if offset is None:
        offset = 0
        h = [-1, -1, -1, offset]
    else:
        h = [-1, offset, -1, _hierarchy[offset][-1]]
        _hierarchy[offset][0] = offset + 1
    child = findContour(childSrc, [h], edgePix=255 * (not edgePix))
    # print(len(child[0]))
    # hierarchy_, contours_ = findContour(nextSrc, [*_hierarchy, h], contours, offset=offset + 1)
    # return hierarchy_, contours_
