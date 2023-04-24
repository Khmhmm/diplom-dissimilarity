import pickle
from glob import glob
from copy import deepcopy as dcpy
from sklearn import svm
import cv2
import numpy as np

import dataset_extractor

OUTPUT_PATH = 'svm.pickle'
TRAIN_DIR = 'assets/'
TRAIN_SIZE = (256, 256)
EPOCH_NUM = 10

# Dataset extraction
print('Extracting data...')
trainset, labelset = dataset_extractor.extract_apples_and_tomatoes()

# Learning
print('\nLearning SVM...')
clf = svm.SVC()
for _ in range(EPOCH_NUM):
    clf = clf.fit(trainset, labelset)
# Test fitting
print('Predict:', clf.predict([trainset[0], trainset[-1]]), 'actual:', labelset[0], labelset[-1])
# Save
with open(OUTPUT_PATH, 'wb') as handle:
    pickle.dump(clf, handle)
