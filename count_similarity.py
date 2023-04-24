# base
import argparse
import os.path as osp
# external
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, match_template
from skimage.filters import prewitt_h, prewitt_v
import matplotlib.pyplot as plt
# local imports
import util

ASSETS_FOLDER = 'assets'
RESULTS_FOLDER = 'results'

def setup_argparser():
    parser = argparse.ArgumentParser(
                prog='Similarity calcer',
                description='Counts similarity'
    )
    parser.add_argument('-f', '--filename')

    return parser


def eq_pixels(p1, p2) -> bool:
    return p1[0] == p2[0] and p1[1] == p2[1] and p1[2] == p2[2]


def calc_bypixel(asset_file: np.ndarray, res_file: np.ndarray) -> float:
    # rows * cols * color_channels
    pixels = asset_file.shape[0] * asset_file.shape[1] * asset_file.shape[2]
    notsim_pixels = 0.

    for i in range(asset_file.shape[0]):
        for j in range(asset_file.shape[1]):
            if not eq_pixels(asset_file[i, j], res_file[i, j]):
                notsim_pixels += 1.

    return 1. - notsim_pixels / pixels



def glcm_extract(img):
    return graycomatrix(img, [1, 2], [0, np.pi/2.], levels=256, symmetric=False)


def prewitt_extract(img):
    return np.vstack((prewitt_h(img), prewitt_v(img)))


def draw_features(asset_file, res_file, asset_file_features, res_file_features):
    plt.subplot(2, 2, 1).imshow(asset_file)
    plt.subplot(2, 2, 2).imshow(res_file)
    plt.subplot(2, 2, 3).imshow(asset_file_features, cmap='gray')
    plt.subplot(2, 2, 4).imshow(res_file_features, cmap='gray')
    plt.show()

'''
Вместо сравнения картинок сделать сравнение фич
https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.greycoprops
GLCM features
entropy feature
'''
def calc_cosine_similarity(asset_file: np.ndarray, res_file: np.ndarray) -> float:
    '''
    idx sequence is:
    contrast, dissimilarity, homogeneity, ASM, energy, correlation
    '''
    # asset_file_features = glcm_extract(
    #     cv2.cvtColor(asset_file, cv2.COLOR_BGR2GRAY)
    # )
    # res_file_features = glcm_extract(
    #     cv2.cvtColor(res_file, cv2.COLOR_BGR2GRAY)
    # )
    asset_file_features = prewitt_extract(
        cv2.cvtColor(asset_file, cv2.COLOR_BGR2GRAY)
    )
    res_file_features = prewitt_extract(
        cv2.cvtColor(res_file, cv2.COLOR_BGR2GRAY)
    )
    # draw_features(asset_file, res_file, asset_file_features, res_file_features)
    return np.dot(asset_file_features.flatten(), res_file_features.flatten()) / (np.linalg.norm(asset_file_features.flatten()) * np.linalg.norm(res_file_features.flatten()))



def calc_template_matching_score(asset_file: np.ndarray, res_file: np.ndarray) -> float:
    match = cv2.cvtColor(asset_file, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(res_file, cv2.COLOR_BGR2GRAY)
    res = match_template(match, template)
    return res[0, 0]


def calc_similarity_pipeline(filename):
    asset_file_path = osp.join('.', osp.join(ASSETS_FOLDER, filename))
    res_file_path = osp.join('.', osp.join(RESULTS_FOLDER, filename))


    asset_file = np.array(cv2.imread(asset_file_path, cv2.IMREAD_COLOR), dtype=np.uint8)
    res_file = np.array(cv2.imread(res_file_path, cv2.IMREAD_COLOR), dtype=np.uint8)

    assert asset_file.shape == res_file.shape

    bypix_sim = calc_bypixel(asset_file, res_file)
    cosine_sim = calc_cosine_similarity(asset_file, res_file)
    template_sim = calc_template_matching_score(asset_file, res_file)

    return bypix_sim, cosine_sim, template_sim


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    sims = calc_similarity_pipeline(args.filename)

    print('Bypix sim:', sims[0])
    print('Cosine sim:', sims[1])
    print('Template matching score', sims[2])
