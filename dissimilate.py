# base
import argparse
import os.path as osp
from glob import glob
from copy import deepcopy as dcpy
import math
# external
import cv2
import numpy as np
from scipy.optimize import curve_fit
from skimage.feature import corner_fast, corner_peaks
from skimage.filters import prewitt_h, prewitt_v
import matplotlib.pyplot as plt
from random import random
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


def pts_eq(p1, p2):
    return p1[1] == p2[1] and p1[0] == p2[0]


def nearest_points(p, points):
    mini, mindistance = 0, BIGGEST_FLOAT
    premini, premindistance = 0, mindistance
    for i, ip in enumerate(points):
        if pts_eq(ip, p):
            continue
        dst = get_distance(ip, p)
        if dst < mindistance:
            premini = mini
            premindistance = mindistance
            mini = i
            mindistance = dst
    return points[mini], points[premini]


def in_ltrb_x(px, ltrb):
    return px >= ltrb[1] and px <= ltrb[3]


def in_ltrb_y(py, ltrb):
    return py >= ltrb[0] and py <= ltrb[2]


def in_ltrb_predicate(p, ltrb):
    return in_ltrb_y(p[0], ltrb) and in_ltrb_x(p[1], ltrb)


def draw_edge(img, p1, p2, coeffs, roi_ltrb, color, thickness, points_num=10):
    if abs(p1[1] - p2[1]) <= 1e-3:
        return img

    p2_path_x = np.array(list( filter(lambda x: in_ltrb_x(x, roi_ltrb), np.arange(p1[1], p2[1], abs(p1[1] - p2[1]) * 1. / points_num)) ))
    p2_path_y = [fit_polynom(x, *coeffs) for x in p2_path_x]

    for i in range(1, len(p2_path_y)):
        prev_px, prev_py = int(p2_path_x[i-1]), int(p2_path_y[i-1])
        px, py = int(p2_path_x[i]), int(p2_path_y[i])
        img = cv2.line(img, (px, py), (prev_px, prev_py), color, thickness=thickness)
    return img


def fit_polynom(x, a, b, c):
    return a * x**2 + b * x + c


def fit_edge(img, p1, p2, p3, roi_ltrb, color, thickness=3):
    ps_x = [p1[1], p2[1], p3[1]]
    ps_y = [p1[0], p2[0], p3[0]]
    popt, pcov = curve_fit(fit_polynom, ps_x, ps_y)
    # random_deviation = [random() * 0.02 * coeff for coeff in popt]
    random_deviation = [coeff for coeff in popt]
    random_deviation[0] *= (1. + 1e-3)

    img = draw_edge(img, p1, p2, random_deviation, roi_ltrb, color, thickness, 20)
    img = draw_edge(img, p1, p3, random_deviation, roi_ltrb, color, thickness, 20)
    img = draw_edge(img, p2, p3, random_deviation, roi_ltrb, color, thickness, 20)

    return img



def dissimilate_image(img: np.ndarray, kernel=[9, 9]):
    hmax = img.shape[0]
    corners = corner_peaks(corner_fast(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), n=8, threshold=0.015), min_distance=1)
    original_img = dcpy(img)
    for p in corners:
        px = p[1]
        py = p[0]
        # img = cv2.circle(img, (px, py), 5, (0, 255, 0), 2)

        p2, p3 = nearest_points(p, corners)
        p2x = p2[1]
        p2y = p2[0]
        p3x = p3[1]
        p3y = p3[0]

        ky1, ky2, kx1, kx2 = py - 1, py + 1, px - 1, px + 1
        roi = img[ky1:ky2, kx1:kx2]
        mean_color = calc_color(roi)

        # TODO: interpolate
        # img = fit_edge(img, p, p2, p3, [ky1, kx1, ky2, kx2], mean_color)
        img = fit_edge(img, p, p2, p3, [ky1, kx1, ky2, kx2], mean_color, thickness=7)

        # img = cv2.line(img, (px, py), (np.clip(p2x, kx1, kx2), np.clip(p2y, ky1, ky2)), mean_color, thickness=3)
        # img[py, px] = mean_color

    img = cv2.addWeighted(original_img, 0.5, img, 0.5, 0)

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
