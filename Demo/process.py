from npcv import *

img = np.load('image.npy')

hierarchy, contour = findContour(np.where(img > 125, 255, 0))
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
    c, pad = contour[k], 2
    min_, max_ = c.min(axis=0), c.max(axis=0)
    im = im[min_[1]:max_[1], min_[0]:max_[0]]
    max_s = max(im.shape)
    pad_ = (max_s - im.shape[0], max_s - im.shape[1])
    im = np.pad(im, ((pad_[0] // 2,) * 2, (pad_[1] // 2,) * 2))
    im = resize(im, 24, 24)
    im = np.pad(im, pad)
    img_list.append(im)
imgs = np.asarray(img_list)
print(imgs.shape)
