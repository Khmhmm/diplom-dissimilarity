import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from net import ConvClassifierNet
from dataset_extractor import Dataset


NUM_EPOCHS = 2
SAVE_PATH = 'apples_cucumbers.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    for i, inputs_labels in enumerate(zip(ds.get_inputs(), ds.get_labels()), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs_labels
        if device != 'cpu':
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            loss_scatter.append(running_loss / 1000)
            epoch_scatter.append(epoch * len(ds.get_labels()) + i / len(ds.get_labels()))
            running_loss = 0.0

    print(f'Save epoch {epoch} to {str(epoch) + "_" + SAVE_PATH}')
    torch.save(cnn.state_dict(), str(epoch) + "_" + SAVE_PATH)

print('Finished Training')
torch.save(cnn.state_dict(), SAVE_PATH)

print(cnn(all_inputs[-1]), all_labels[-1])

print(f'Saved pth to {SAVE_PATH}. Show loss graphic...')
print(epoch_scatter)
print('--')
print(loss_scatter)
plt.plot(epoch_scatter, loss_scatter)
plt.show()
