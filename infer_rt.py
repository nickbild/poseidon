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
from train import Net


trained_model = "poseidon_5_97.67441860465117.model"
img = "data/test/0/4_18.wav"


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Load the saved model.
checkpoint = torch.load(trained_model)
model = Net()
model.load_state_dict(checkpoint)
model.eval()

# Load returns a tensor with the sound data and the sampling frequency.
sound = torchaudio.load(img, out = None, normalization = True)
mixer = torchaudio.transforms.DownmixMono(channels_first=True)
soundData = mixer(sound[0]) # Mono

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

data = soundFormatted
# Add an extra batch dimension since pytorch treats all images as batches.
data = data.unsqueeze_(0)

data = data.to(device)
output = model(data)
output = output.permute(1, 0, 2)
pred = output.max(2)[1].item()
print(pred)
