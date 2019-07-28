import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import IPython.display as ipd
import re


# Play wav file.
# ipd.Audio('path/to/file.wav')

class CommandDataset(Dataset):
    def __init__(self, file_path):
        self.file_names = []
        self.labels = []

        for label in os.listdir(file_path):
            if not re.search("^\.", label):
                for file in os.listdir(file_path + "/" + label):
                    if not re.search("^\.", file):
                        self.labels.append(int(label))
                        self.file_names.append(file_path + "/" + label + "/" + file)

        self.mixer = torchaudio.transforms.DownmixMono()

    def __getitem__(self, index):
        path = self.file_names[index]

        # Load returns a tensor with the sound data and the sampling frequency.
        sound = torchaudio.load(path, out = None, normalization = True)
        soundData = self.mixer(sound[0]) # Mono

        # Audio is 44100 Hz, so 30K samples = 0.68s
        # Downsample 1/3rd = ~ 2s audio time.
        soundFormatted = torch.zeros([30000, 1])
        soundFormatted[:30000] = soundData[::3]
        soundFormatted = soundFormatted.permute(1, 0)

        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim = 2)


def train(model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_() # True for training.
        output = model(data)
        output = output.permute(1, 0, 2)
        loss = F.nll_loss(output[0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]
        correct += pred.eq(target).cpu().sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    print("Device: {}".format(device))

    train_set = CommandDataset("data/train")
    test_set = CommandDataset("data/test")
    print("Train set size: " + str(len(train_set)))
    print("Test set size: " + str(len(test_set)))

    if device == 'cuda':
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = False, **kwargs)

    model = Net()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

    log_interval = 20
    for epoch in range(1, 41):
        scheduler.step()
        train(model, epoch)
        test(model, epoch)