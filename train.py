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
import re


#import IPython.display as ipd
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

        self.mixer = torchaudio.transforms.DownmixMono(channels_first=True)

    def __getitem__(self, index):
        path = self.file_names[index]

        # Load returns a tensor with the sound data and the sampling frequency.
        sound = torchaudio.load(path, out = None, normalization = True)
        soundData = self.mixer(sound[0]) # Mono

        # Pad tensor for minimum size of 88,064 frames (2s, 44,100 Hz).
        if soundData.shape[1] < 88064:
            padded = torch.zeros(1, 88064)
            padded[:, :soundData.shape[1]] = soundData
            soundData = padded

        soundData = soundData.view(88064, -1)

        # Audio is 44100 Hz, so 29,355 samples = 0.66s
        # Downsample 1/3rd = 2s audio time.
        soundFormatted = torch.zeros([29355, 1])
        soundFormatted[:29355] = soundData[::3]
        soundFormatted = soundFormatted.permute(1, 0)

        return soundFormatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(0.35)

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
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.dropout(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 2)


def train(model, epoch):
    model.train()
    train_acc = 0

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

        _, prediction = torch.max(output.data, 2)
        train_acc += torch.sum(prediction == target.data)

        train_acc_pct = "{0:.2f}".format((float(train_acc) / float(len(train_loader.dataset))) * 100.0)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Acc: {} ({}%)'.format(
            epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
            100. * (batch_idx+1) / len(train_loader), loss, train_acc, train_acc_pct))


def test(model, epoch, max_accuracy):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1]
        correct += pred.eq(target).cpu().sum().item()

    if correct >= max_accuracy:
       max_accuracy = correct
       save_models(epoch, 100. * correct / len(test_loader.dataset))

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return max_accuracy


def save_models(epoch, test_acc_pct):
    torch.save(model.state_dict(), "poseidon_{}_{}.model".format(epoch, test_acc_pct))
    print("Model saved.")


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

    optimizer = optim.Adam(model.parameters(), lr = 0.000005, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 40, gamma = 0.1)

    max_accuracy = 0
    for epoch in range(1, 50001):
        scheduler.step()
        train(model, epoch)
        max_accuracy = test(model, epoch, max_accuracy)
