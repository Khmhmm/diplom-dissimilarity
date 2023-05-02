import cv2
import numpy as np
import torch
from glob import glob
import os.path as osp

APPLES_DIR = '/media/matvey/6bf4fb31-7931-4e05-80ce-fca506bac8451/git/diplom/datasets/apples'
CUCUMBERS_DIR = '/media/matvey/6bf4fb31-7931-4e05-80ce-fca506bac8451/git/diplom/datasets/cucumbers'

TRAIN_SIZE = (256, 256)

class Dataset:
    def __init__(self, cl1_dir=APPLES_DIR, cl2_dir=CUCUMBERS_DIR, batch_size=16):
        # do something
        self.cl1_dir = cl1_dir
        self.cl2_dir = cl2_dir
        self.batch_size = batch_size

        inputs, labels = self._process_dirs()
        self.inputs, self.labels = self._batch_slice(inputs, labels)

    def _process_dirs(self):
        def extract_part(dirp, class_num=0):
            li = []
            img_ps = glob(osp.join(osp.join(dirp, 'train'), '*.jpeg')) + glob(osp.join(osp.join(dirp, 'train'), '*.jpg')) + glob(osp.join(osp.join(dirp, 'train'), '*.png'))
            for img_path in img_ps:
                base_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), TRAIN_SIZE)
                normalized_img = base_img / 256
                normalized_img = normalized_img.transpose((2, 0, 1))
                li.append(normalized_img)
            return li, [class_num] * len(li)

        apples, app_labels = extract_part(self.cl1_dir, 0)
        cucumbers, cucumbers_labels = extract_part(self.cl2_dir, 1)

        return apples + cucumbers, app_labels + cucumbers_labels

    def _batch_slice(self, inputs, labels):
        raw_inputs = np.array(inputs, dtype=np.float32)
        raw_labels = np.array(labels, dtype=np.int64)

        batched_inputs = []
        batched_labels = []

        cnt = 0
        new_batch_inputs = []
        new_batch_labels = []
        for i in range(raw_inputs.shape[0]):
            if (cnt != 0 and cnt % self.batch_size == 0) or i == raw_inputs.shape[0] - 1:
                cnt = 0
                batched_inputs.append(torch.from_numpy(np.array(new_batch_inputs)))
                new_batch_inputs = []
                batched_labels.append(torch.from_numpy(np.array(new_batch_labels)))
                new_batch_labels = []
                continue

            new_batch_inputs.append(raw_inputs[i])
            new_batch_labels.append(raw_labels[i])
            cnt += 1

        return batched_inputs, batched_labels


    def get_classes_num(self):
        return 2

    def get_inputs(self):
        '''
        Returns inputs with shape: [batches_num, channels_num, width, height]
        '''
        return self.inputs

    def get_labels(self):
        '''
        Returns labels like: [batch_num, class_label]
        '''
        return self.labels
