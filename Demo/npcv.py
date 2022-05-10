import numpy as np


def bordering(_i, _j, xSize: tuple, ySize: tuple, connect=8):
    assert connect in (4, 8)
    mi, mj, pi, pj = _i - 1, _j - 1, _i + 1, _j + 1
    if connect == 8:
        rVal = [(pi, _j), (pi, pj), (_i, pj), (mi, pj), (mi, _j), (mi, mj), (_i, mj), (pi, mj)]
    else:
        rVal = [(pi, _j), (_i, pj), (mi, _j), (_i, mj)]
    rValNew = []
    for r in rVal:
        if ySize[0] <= r[1] < ySize[1] and xSize[0] <= r[0] < xSize[1]: rValNew.append(r)
    return tuple(rValNew)


def floodFill(src, bound, wallCol, defCol):
    mask = np.zeros_like(src, dtype=bool)
    bound = (bound[0][0] - 1, bound[0][1] + 2), (bound[1][0] - 1, bound[1][1] + 2)
    mask[bound[1][0]:bound[1][1], bound[0][0]:bound[0][1]] = True
    mask[wall := src == wallCol] = False
    start = np.where((src == defCol) & mask)
    stack = {(start[1][0], start[0][0])}
    while stack:
        x, y = stack.pop()
        if mask[y, x]:
            mask[y, x] = False
            stack.update(bordering(x, y, bound[0], bound[1], connect=4))
    mask[wall] = True
    return mask


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
    elif lineWidth in (-1, -2):
        srcCopy = drawContour(np.zeros_like(src), _contour, 255)
        _max, _min = _contour.max(axis=0), _contour.min(axis=0)
        mask = floodFill(srcCopy, ((_min[0], _max[0]), (_min[1], _max[1])), wallCol=255, defCol=0)
        if lineWidth == -2: return mask
        return src * ~mask + mask * np.uint8(col)
    else:
        raise ValueError("lineWidth must be >= -2")


def findContour(src, _hierarchy=None, _contours=None, offset=None, edgePix=None):
    if edgePix is None: edgePix = 0
    if _hierarchy is None: _hierarchy = [[-1, -1, -1, -1]]
    if _contours is None: _contours = [[]]
    notEdgePIx = 255 * (not edgePix)
    edge = src == edgePix
    if np.alltrue(~edge): return _hierarchy, _contours
    start = divmod(np.argmax(edge), src.shape[0])[::-1]
    contours = [start]
    borders = bordering(*start, xSize=(0, src.shape[1]), ySize=(0, src.shape[0]))
    prev = start
    index = 0
    current = borders[index]
    while current != start and index < 8:
        if src[current[::-1]] == edgePix:
            contours.append(current)
            borders = bordering(*current, xSize=(0, src.shape[1]), ySize=(0, src.shape[0]))
            index = borders.index(prev)
            prev = current
            borders = borders[index:] + borders[:index]
            index = 0
        index += 1
        current = borders[index]
    contours = np.array(contours)
    mask = drawContour(src, contours, lineWidth=-2)
    nextSrc, childSrc = src * ~mask + np.uint8(notEdgePIx) * mask, src * mask + np.uint8(edgePix) * ~mask
    if offset is None:
        h = [-1, -1, -1, len(_hierarchy) - 1]
        _hierarchy[-1][-2] = len(_hierarchy)
    else:
        h = [-1, offset, -1, _hierarchy[offset][-1]]
        _hierarchy[offset][0] = len(_hierarchy)
    child = findContour(childSrc, [*_hierarchy, h], [*_contours, contours], edgePix=notEdgePIx)
    sibling = findContour(nextSrc, *child, offset=len(_hierarchy), edgePix=edgePix)
    return np.array(sibling[0]), sibling[1]


def resize(src, x, y):
    y0, x0 = src.shape
    newSrc = np.zeros((y, x), dtype='uint8')
    x1, y1 = x0 / x, y0 / y
    span = np.ceil((x1, y1)).astype(int) + 1
    for j, row in enumerate(newSrc):
        for i, pix in enumerate(row):
            j_, i_ = int(y1 * j), int(x1 * i)
            newSrc[j, i] = src[j_:j_+span[1], i_:i_+span[0]].mean()
    return newSrc
