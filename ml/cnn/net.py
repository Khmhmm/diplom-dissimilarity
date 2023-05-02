import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvClassifierNet(nn.Module):
    def __init__(self, classes=2, conv_kernel=5, img_dim=256):
        super().__init__()
        '''
        Conv operation explanation:
        X = [num_imgs, num_channels, img_dim, img_dim]
        conv = (num_channels, num_feats, conv_kernel)
        conv(X) => [num_imgs, num_feats, img_dim, img_dim]
        * num_imgs can also be considered as batch_size
        '''
        self.cnv1 = nn.Conv2d(3, 6, conv_kernel)
        self.pl = nn.MaxPool2d(2, 2)
        self.cnv2 = nn.Conv2d(6, 16, conv_kernel)
        self.cnv3 = nn.Conv2d(16, 32, conv_kernel)
        # Хардкод лютый
        self.lin1 = nn.Linear(32 * 28 * 28, 120)
        self.lin2 = nn.Linear(120, 64)
        self.lin3 = nn.Linear(64, classes)

    def forward(self, x):
        # convolution
        x = self.pl(F.relu(self.cnv1(x)))
        x = self.pl(F.relu(self.cnv2(x)))
        x = self.pl(F.relu(self.cnv3(x)))
        # 2d array to 1d array
        x = torch.flatten(x, 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        # returns array with normalized confidences, like [0.1, 0.5, ...]
        # each number is confidence in class, for example, x[0] is a confidence in 0-th class
        return x
