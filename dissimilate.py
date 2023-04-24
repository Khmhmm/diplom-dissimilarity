# base
import argparse
import os.path as osp
from glob import glob
from copy import deepcopy as dcpy
import math
# external
import cv2
import numpy as np
from skimage.feature import corner_fast, corner_peaks
from skimage.filters import prewitt_h, prewitt_v
import matplotlib.pyplot as plt
# local imports


ASSETS_FOLDER = 'assets'
RESULTS_FOLDER = 'results'
BIGGEST_FLOAT = math.exp(math.pi**(1**100))


def calc_color(roi):
    lt = roi[0, 0]/4.
    rt = roi[0, -1]/4.
    lb = roi[-1, 0]/4.
    rb = roi[-1, -1]/4.
    return lt + rt + lb + rb


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]))


def nearest_point(p, points):
    mini, mindistance = 0, BIGGEST_FLOAT
    for i, ip in enumerate(points):
        if ip.all() == p.all():
            continue
        dst = get_distance(ip, p)
        if dst < mindistance:
            mini = i
            mindistance = dst
    return points[mini]


def dissimilate_image(img: np.ndarray, kernel=[1, 1]):
    hmax = img.shape[0]
    corners = corner_peaks(corner_fast(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), n=8, threshold=0.15), min_distance=1)
    for p in corners:
        px = p[1]
        py = p[0]
        # img = cv2.circle(img, (px, py), 5, (0, 255, 0), 2)

        p2 = nearest_point(p, corners)
        p2x = p2[1]
        p2y = p2[0]

        ky1, ky2, kx1, kx2 = py - kernel[0], py + kernel[0], px - kernel[1], px + kernel[1]
        roi = img[ky1:ky2, kx1:kx2]
        mean_color = calc_color(roi)

        # TODO: interpolate
        img = cv2.line(img, (px, py), (np.clip(p2x, kx1, kx2), np.clip(p2y, ky1, ky2)), mean_color, thickness=3)
        # img[py, px] = mean_color
    return img


def process_files():
    files = glob(osp.join('.', osp.join(ASSETS_FOLDER, '*.png'))) + glob(osp.join('.', osp.join(ASSETS_FOLDER, '*.jpg')))

    for f in files:
        print('Processing image:', f)
        res_file = np.array(cv2.imread(f, cv2.IMREAD_COLOR), dtype=np.uint8)
        res_file = dissimilate_image(res_file)
        cv2.imwrite(f.replace(ASSETS_FOLDER, RESULTS_FOLDER), res_file)


if __name__ == '__main__':
    process_files()
