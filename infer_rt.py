import torch
import torchaudio
from train import Net
import pyaudio
import wave


trained_model = "poseidon_5_97.1875.model"
wav_file = "rt_audio.wav" #"data/test/0/4_18.wav"

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

lookup = []
lookup.append("Hey, Google")
lookup.append("Alexa")
lookup.append("Background Noise")

# Load the saved model.
checkpoint = torch.load(trained_model)
model = Net()
model.load_state_dict(checkpoint)
model.eval()


def main():
    while True:
        record_wav()
        data = load_img_to_tensor(wav_file)

        prediction, score = predict_class(data)
        print("{} {}".format(lookup[prediction], score))


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
    score = output.max(2)[0].item()

    return pred, score


def record_wav():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wav_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))


if __name__ == "__main__":
    main()
