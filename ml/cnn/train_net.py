import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

import os.path as osp

from net import ConvClassifierNet
from dataset_extractor import Dataset


NUM_EPOCHS = 10000
SAVE_DIR = 'weights'
SAVE_PATH = 'apples_cucumbers.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f'Net will be trained on {device}')

print('Loading dataset...')
ds = Dataset()

print('Setting up network')
cnn = ConvClassifierNet(classes=ds.get_classes_num())
cnn.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
loss_scatter = []
epoch_scatter = []

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    all_inputs, all_labels = ds.get_inputs(), ds.get_labels()
    running_loss = 0.0
    last_loss = 0.
    for i, inputs_labels in enumerate(zip(ds.get_inputs(), ds.get_labels()), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs_labels
        if device != 'cpu':
            inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # score loss for each 10th minibatch
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10
            running_loss = 0.0

    if epoch % (NUM_EPOCHS / 10) == NUM_EPOCHS / 10 - 1:
        print(f'Save epoch {epoch} to {str(epoch) + "_" + SAVE_PATH}')
        torch.save(cnn.state_dict(), osp.join(SAVE_DIR, str(epoch) + "_" + SAVE_PATH))

    print(f'[{epoch + 1} ep.] loss: {last_loss:.3f}')
    loss_scatter.append(last_loss)
    epoch_scatter.append(epoch)

print('Finished Training')
torch.save(cnn.state_dict(), osp.join(SAVE_DIR, 'last_' + SAVE_PATH))

print(cnn(all_inputs[0].to(device)), all_labels[0])
print(cnn(all_inputs[-1].to(device)), all_labels[-1])

print(f'Saved pth to {SAVE_PATH}. Show loss graphic...')
plt.plot(epoch_scatter, loss_scatter)
plt.show()
