import pickle
from glob import glob
from copy import deepcopy as dcpy
from sklearn import svm, datasets
import cv2
import numpy as np

MODEL_PATH = 'svm.pickle'
TEST_DIR = 'test_assets/'
TEST_POISON_DIR = 'test_poison/'
TRAIN_SIZE = (256, 256)

def extract_ds(dir_path):
    li = []
    img_ps = glob(dir_path + '*.png') + glob(dir_path + '*.jpg')+ glob(dir_path + '*.jpeg')
    for img_path in img_ps:
        li.append(
            np.resize(
                cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), TRAIN_SIZE),
                (TRAIN_SIZE[0] * TRAIN_SIZE[1])
            )
        )
    return li

test = list()
predictions = list()

poisoned = list()
poisoned_predictions = list()

with open(MODEL_PATH, 'rb') as handle:
    model = pickle.load(handle)

test = extract_ds(TEST_DIR)
predictions = model.predict(test)

del test

poisoned = extract_ds(TEST_POISON_DIR)
poisoned_predictions = model.predict(poisoned)

del poisoned

print('Base img:', predictions)
print('Poisoned:', poisoned_predictions)
