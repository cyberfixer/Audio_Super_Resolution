# This file converts low sample rate audio to high sample rate audio using the pretrained model (checkpoints)


import torch
import torch.nn as nn
import os
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Below modules will not be imported without this
import sys
sys.path.append('src/')

from config import CONFIG
from model import SSAR, TUNet
from dataset import CustomDataset, frame, pad

# Global Variables

inputAudioRoot = './data/vctk/8k/'
inputAudio = 'p255/p255_001_mic1.flac'
inputCheckpoint = './checkpoints/Epoch200_loss2242.pt'

outputFolder = './output/'

def getAudio(path):
    low_sig, low_sig_sr = librosa.load(path, sr=None)

    # Upsample Audio
    low_sig = librosa.resample(low_sig, orig_sr=low_sig_sr, target_sr=16000)
    
    if len(low_sig) < CONFIG.DATA.window_size:
        # padding if the window is longer than the signal
        low_sig = pad(low_sig, CONFIG.DATA.window_size)


    x = frame(low_sig, CONFIG.DATA.window_size, CONFIG.DATA.stride)[:, np.newaxis, :]
    return torch.tensor(x)

def combineWindows(verticalSignal): # TODO: combine overlapping windows
    horizontalSignal = torch.empty(0)
    for i in verticalSignal.squeeze(1):
        horizontalSignal = torch.cat((horizontalSignal,i))
    return horizontalSignal

def main():

    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = TUNet().to(device)
    checkpoint = torch.load(inputCheckpoint, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

    # Get the source audio
    lowSignal = getAudio(os.path.join(inputAudioRoot, inputAudio))
    lowSignal = lowSignal.to(device)

    # Set model to evaluation mode
    model.eval()

    with torch.inference_mode():
        # Predict signal using the model
        predictedSignal = model(lowSignal)

        predictedSignal = predictedSignal.detach().cpu()
        # Combine all audio windows to be 1D array
        horizontalPredSig = combineWindows(predictedSignal)
        horizontalPredSig = horizontalPredSig.numpy()

        # Output the audio
        sf.write(os.path.join(outputFolder,'predicted_p255_001_mic1.flac') , data=horizontalPredSig,samplerate=16000, format='flac')

        # Visualize it

        # Compute the STFT (Short Time Fourier Transform)
        stft = librosa.stft(horizontalPredSig)

        # Convert the STFT matrix to dB scale
        stft_db = librosa.amplitude_to_db(abs(stft))

        # TODO: Add Subplots for comparison
        # Plot the spectrogram
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(stft_db, x_axis='time', y_axis='linear')
        plt.colorbar()
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()
    
if __name__ == '__main__':
    main()
    
