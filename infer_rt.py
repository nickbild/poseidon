import torch
import torchaudio
from train import Net


trained_model = "poseidon_5_97.67441860465117.model"
image_file = "data/test/0/4_18.wav"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Load the saved model.
checkpoint = torch.load(trained_model)
model = Net()
model.load_state_dict(checkpoint)
model.eval()


def main():
    data = load_img_to_tensor(image_file)
    prediction = predict_class(data)

    print(prediction)


def load_img_to_tensor(img):
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

    # Add an extra batch dimension since pytorch treats all images as batches.
    soundFormatted = soundFormatted.unsqueeze_(0)

    return soundFormatted


def predict_class(data):
    data = data.to(device)
    output = model(data)
    output = output.permute(1, 0, 2)
    pred = output.max(2)[1].item()

    return pred


if __name__ == "__main__":
    main()
