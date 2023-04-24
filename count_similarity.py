import argparse
import os.path as osp
import cv2
import numpy as np

ASSETS_FOLDER = 'assets'
RESULTS_FOLDER = 'results'

def setup_argparser():
    parser = argparse.ArgumentParser(
                prog='Similarity counter',
                description='Counts similarity'
    )
    parser.add_argument('-f', '--filename')

    return parser


def eq_pixels(p1, p2) -> bool:
    return p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]


def count_bypixel(asset_file: np.ndarray, res_file: np.ndarray) -> float:
    # rows * cols * color_channels
    pixels = asset_file.shape[0] * asset_file.shape[1] * asset_file.shape[2]
    notsim_pixels = 0.

    for i in range(asset_file.shape[0]):
        for j in range(asset_file.shape[1]):
            if not eq_pixels(asset_file[i, j], res_file[i, j]):
                notsim_pixels += 1.

    return 1. - notsim_pixels / pixels


def count_cosine_similarity(asset_file: np.ndarray, res_file: np.ndarray) -> float:
    return 1. - np.dot(asset_file.flatten(), res_file.flatten()) / (np.linalg.norm(asset_file.flatten()) * np.linalg.norm(res_file.flatten()))



if __name__ == '__main__':
    args = setup_argparser().parse_args()
    asset_file_path = osp.join('.', osp.join(ASSETS_FOLDER, args.filename))
    res_file_path = osp.join('.', osp.join(RESULTS_FOLDER, args.filename))


    asset_file = np.array(cv2.imread(asset_file_path, cv2.IMREAD_COLOR), dtype=np.uint8)
    res_file = np.array(cv2.imread(res_file_path, cv2.IMREAD_COLOR), dtype=np.uint8)

    assert asset_file.shape == res_file.shape

    bypix_sim = count_bypixel(asset_file, res_file)
    print('By pixel similarity:', bypix_sim)

    cosine_sim = count_cosine_similarity(asset_file, res_file)
    print('By cosine similarity:', cosine_sim)
