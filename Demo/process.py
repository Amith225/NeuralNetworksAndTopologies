import cv2
import os

import numpy as np

from npcv import *

img = cv2.imread(os.path.dirname(os.path.dirname(__file__)) + '/image.png', cv2.IMREAD_UNCHANGED)
a1 = ~img[:, :, 3]
img = cv2.add(cv2.merge([a1, a1, a1, a1]), img)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY).astype('uint8')

hierarchy, contour = findContour(cv2.threshold(img, 225 / 2, 255, cv2.THRESH_BINARY)[1])
imgs = {}
roots = [0]
for i, (c, h) in enumerate(zip(contour[1:], hierarchy[1:])):
    if h[-1] in roots:
        im = drawContour(np.zeros_like(img), c, 255, -1)
        child = h[-2]
        if hierarchy[child][-2] != -1: roots.append(child)
        while child != -1:
            im = drawContour(im, contour[child], 0, -1)
            child = hierarchy[child][0]
        imgs[i + 1] = im

img_list = []
for k, im in imgs.items():
    c, pad = contour[k], 20
    min_, max_ = c.min(axis=0), c.max(axis=0)
    im = im[min_[1]:max_[1], min_[0]:max_[0]]
    max_s = max(im.shape)
    pad_ = (max_s - im.shape[0], max_s - im.shape[1])
    im = np.pad(im, ((pad_[0] // 2,) * 2, (pad_[1] // 2,) * 2))
    im = cv2.resize(im, (260, 260))
    im = np.pad(im, pad)
    cv2.imshow('', im), cv2.waitKey(0), cv2.destroyAllWindows(), cv2.waitKey(1)
    img_list.append(im)
