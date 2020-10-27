import time
import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans

COLOR_RED = [0, 0, 255]
OPACITY = 0.5

def mask2Pointcloud(mask):
    (h, w) = mask.shape
    xv, yv = np.meshgrid(np.arange(h), np.arange(w))
    coordinate = np.stack((yv, xv), axis=2)
    pointcloud = coordinate[np.array(mask, dtype=bool)]
    return pointcloud

def kmeans(mask):
    poinclount = mask2Pointcloud(mask)
    kstart = time.time()
    kmeans = MiniBatchKMeans(n_clusters=2).fit(poinclount)
    print('KMeans estimated time: {:.4f}'.format(time.time()-kstart))
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    for ([i, j], l) in zip(poinclount, kmeans.labels_):
        if l == 0:
            mask[i, j] = [1, 0, 0]
        else:
            mask[i, j] = [0, 1, 0]
    return mask

def get_mask_by_polygon(img, polygon):
    mask = np.zeros(img.shape, dtype=img.dtype)
    mask = cv.fillPoly(mask, [polygon], COLOR_RED)
    red_mask = np.full(img.shape, COLOR_RED, dtype=img.dtype)
    mask = np.equal(mask, red_mask).all(axis=2)
    return mask


def draw_polygon(img, polygon):
    mask = get_mask_by_polygon(img, polygon)
    img[mask] = img[mask] * (1-OPACITY) + np.array(COLOR_RED) * OPACITY
    return img
