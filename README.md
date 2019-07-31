# Poseidon

Rename your smart speaker.

Coming soon!

## AWS

```
sudo apt-get install sox libsox-dev

git clone --branch v0.2.0 https://github.com/pytorch/audio.git
cd audio
git checkout d92de5b
python3 setup.py install

cd ..
git clone https://github.com/nickbild/poseidon.git
cd poseidon
source pytorch_p36
python3 train.py
```
