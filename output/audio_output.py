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
inputTargetAudioRoot = './data/vctk/16k/'
inputAudio = 'p255/p255_001_mic1.flac'
inputCheckpoint = './checkpoints/Epoch200_loss2242.pt'

outputFolder = './output/'

def getAudio(relpath):
    low_sig, low_sig_sr = librosa.load(os.path.join(inputAudioRoot + relpath), sr=None)
    high_sig, _ = librosa.load(os.path.join(inputTargetAudioRoot + relpath), sr=None)

    # Upsample Audio
    low_sig_upsampled = librosa.resample(low_sig, orig_sr=low_sig_sr, target_sr=16000)
    
    if len(low_sig_upsampled) < CONFIG.DATA.window_size:
        # padding if the window is longer than the signal
        low_sig_upsampled = pad(low_sig_upsampled, CONFIG.DATA.window_size)


    x = frame(low_sig_upsampled, CONFIG.DATA.window_size, CONFIG.DATA.stride)[:, np.newaxis, :]
    return torch.tensor(x), low_sig, high_sig

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
    windowedLowSignal, lowSignal, highSignal = getAudio(inputAudio)
    windowedLowSignal = windowedLowSignal.to(device)

    # Set model to evaluation mode
    model.eval()

    with torch.inference_mode():
        # Predict signal using the model
        predictedSignal = model(windowedLowSignal)

        predictedSignal = predictedSignal.detach().cpu()
        # Combine all audio windows to be 1D array
        horizontalPredSig = combineWindows(predictedSignal)
        horizontalPredSig = horizontalPredSig.numpy()

        # Output the audio
        sf.write(os.path.join(outputFolder,'predicted_p255_001_mic1.flac') , data=horizontalPredSig,samplerate=16000, format='flac')

        # Visualize it
        # TODO: MAKE THE SAMPLE RATE OF EACH SPECSHOW FUNCTION DYNAMIC
        fig = plt.figure(figsize=(15, 8))

        ''' LOW SIGNAL SUBPLOT '''
        # Compute the STFT (Short Time Fourier Transform)
        stft = librosa.stft(lowSignal)
        # Convert the STFT matrix to dB scale
        stft_db = librosa.amplitude_to_db(abs(stft))

        # Plot the spectrogram
        fig.add_subplot(1,3,1)
        librosa.display.specshow(stft_db, x_axis='time', y_axis='linear', sr=8000)
        plt.colorbar()
        plt.title('Original')
        plt.ylim(top=8000)
        plt.tight_layout()

        ''' PREDICTED SIGNAL SUBPLOT '''
        # Compute the STFT (Short Time Fourier Transform)
        stft = librosa.stft(horizontalPredSig)
        # Convert the STFT matrix to dB scale
        stft_db = librosa.amplitude_to_db(abs(stft))

        # Plot the spectrogram
        fig.add_subplot(1,3,2)
        librosa.display.specshow(stft_db, x_axis='time', y_axis='linear', sr=16000)
        plt.colorbar()
        plt.title('Predicted')
        plt.tight_layout()

        ''' HIGH SIGNAL SUBPLOT '''
        # Compute the STFT (Short Time Fourier Transform)
        stft = librosa.stft(highSignal)
        # Convert the STFT matrix to dB scale
        stft_db = librosa.amplitude_to_db(abs(stft))

        # Plot the spectrogram
        fig.add_subplot(1,3,3)
        librosa.display.specshow(stft_db, x_axis='time', y_axis='linear', sr=16000)
        plt.colorbar()
        plt.title('Target')
        plt.tight_layout()

        plt.subplots_adjust(wspace=0.15)
        plt.show()
    
if __name__ == '__main__':
    main()
    
