from glob import glob
import cv2
import numpy as np
from sklearn import datasets

TRAIN_SIZE = (256, 256)

APPLES_AND_TOMATOES = '../../../datasets/apple_tomatoes/'

def extract_digits():
    ds = datasets.load_digits()

    trainset = ds.images.tolist()
    labelset = ds.target

    for i in range(len(trainset)):
        img = np.array(trainset[i])
        img = cv2.resize(img, TRAIN_SIZE)
        _, img = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)
        trainset[i] = img
        # cv2.imwrite(f'{TRAIN_DIR}num_{labelset[i]}_{i}.jpg', img)
    trainset = np.array(trainset)
    trainset = np.reshape(trainset, (trainset.shape[0], TRAIN_SIZE[0] * TRAIN_SIZE[1]))

    return trainset, labelset

def extract_apples_and_tomatoes():
    def extract_part(dirn, class_num=0):
        li = []
        img_ps = glob(APPLES_AND_TOMATOES + dirn + '*.jpeg')
        for img_path in img_ps:
            li.append(
                np.resize(
                    cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), TRAIN_SIZE),
                    (TRAIN_SIZE[0] * TRAIN_SIZE[1])
                )
            )
        return li, [class_num] * len(li)

    apples, app_labels = extract_part('apples/', 0)
    toms, tom_labels = extract_part('tomatoes/', 1)

    return apples + toms, app_labels + tom_labels
