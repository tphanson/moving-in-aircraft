import numpy as np
import cv2 as cv

COLOR_RED = [0, 0, 255]
OPACITY = 0.5


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
