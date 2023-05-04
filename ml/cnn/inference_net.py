import torch
from torch.nn import functional as F
import cv2
import numpy as np

from net import ConvClassifierNet

WEIGHTS_PATH = 'weights/239_apples_cucumbers.pth'
TRAIN_SIZE = (256, 256)
BASE_IMAGE_PATH = '../../assets/59KQWV3RMNM1.jpg'
DISSIM_IMAGE_PATH = '../../results/59KQWV3RMNM1.jpg'

cnn = ConvClassifierNet()
cnn.load_state_dict(torch.load(WEIGHTS_PATH))

def prepare_image(img_path: str):
    base_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), TRAIN_SIZE)
    normalized_img = base_img / 256
    normalized_img = normalized_img.transpose((2, 0, 1))
    return normalized_img

base_img = prepare_image(BASE_IMAGE_PATH)
dissim_img = prepare_image(DISSIM_IMAGE_PATH)

inp = np.array([base_img, dissim_img], dtype=np.float32)
inp = torch.from_numpy(inp)

outp = cnn(inp)
outp = F.normalize(outp)
print(outp)
