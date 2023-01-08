# This file converts low sample rate audio to high sample rate audio using the pretrained model (checkpoints)

from loss import MRSTFTLossDDP
import torch
import torch.nn as nn

from config import CONFIG
from model import SSAR, TUNet
from dataset import CustomDataset, frame, pad


import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def getAudio(path):
    low_sig, _ = librosa.load(path, sr=None)
    
    if len(low_sig) < CONFIG.DATA.window_size:
        # padding if the window is longer than the signal
        low_sig = pad(low_sig, CONFIG.DATA.window_size)


    x = frame(low_sig, CONFIG.DATA.window_size, CONFIG.DATA.stride)[:, np.newaxis, :]
    return torch.tensor(x)

def combineWindows(verticalSignal):
    horizontalSignal = torch.empty(0)
    for i in verticalSignal.squeeze(1):
        horizontalSignal = torch.cat((horizontalSignal,i))
    return horizontalSignal

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TUNet().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

checkpoint = torch.load('./checkpoints/Epoch700_loss25935.pt',map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
#loss = checkpoint['loss']

lowSignal = getAudio('./data/vctk/8k/p255/p255_001_mic1.flac')
lowSignal = lowSignal.to(device)

model.eval()

with torch.inference_mode():
    predictedSignal = model(lowSignal)

    predictedSignal = predictedSignal.detach().cpu()
    horizontalPredSig = combineWindows(predictedSignal)
    horizontalPredSig = horizontalPredSig.numpy()
    horizontalPredSig = np.pad(horizontalPredSig, horizontalPredSig.size)
    sf.write('predicted_p255_001_mic1.flac', data=horizontalPredSig,samplerate=16000, format='flac')

    # Compute the STFT
    stft = librosa.stft(horizontalPredSig)

    # Convert the STFT matrix to dB scale
    stft_db = librosa.amplitude_to_db(abs(stft))

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(stft_db, x_axis='time', y_axis='linear')
    plt.colorbar()
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    
