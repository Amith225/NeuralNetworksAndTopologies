import numpy as np
import cv2
import os
from cv import *

img = cv2.imread(os.path.dirname(os.path.dirname(__file__)) + '/image.png', cv2.IMREAD_UNCHANGED)
a1 = ~img[:, :, 3]
img = cv2.add(cv2.merge([a1, a1, a1, a1]), img)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY).astype('uint8')

img = cv2.threshold(img, 225 / 2, 255, cv2.THRESH_BINARY)[1]
# contour_, hierarchy_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hierarchy, contours = findContour(img)
print(hierarchy)
exit()
imgs = {}
roots = [0]
for i, (c, h) in enumerate(zip(contour_[1:], hierarchy_[1:])):
    if h[-1] in roots:
        im = cv2.drawContours(np.zeros_like(img), [c], 0, 255, -1)
        child = h[-2]
        if hierarchy_[child][-2] != -1: roots.append(child)
        while child != -1:
            im = cv2.drawContours(im, [contour_[child]], 0, 0, -1)
            child = hierarchy_[child][0]
        imgs[i + 1] = im

img_list = []
for k, im in imgs.items():
    c, pad = contour_[k], 20
    min_, max_ = c.min(axis=0)[0], c.max(axis=0)[0]
    im = im[min_[1]:max_[1], min_[0]:max_[0]]
    max_s = max(im.shape)
    pad = (max_s - im.shape[0] + pad, max_s - im.shape[1] + pad)
    im = np.pad(im, ((pad[0] // 2,) * 2, (pad[1] // 2,) * 2))
    im = cv2.resize(im, (280, 280)).astype('uint8')
    cv2.imshow('', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    img_list.append(im)
