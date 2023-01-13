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
inputPredictedAudioSR = 16000
inputCheckpoint = './checkpoints/Epoch1700_loss2219.pt'

outputFolder = './output/'

def getAudio(relpath):
    low_sig, low_sig_sr = librosa.load(os.path.join(inputAudioRoot + relpath), sr=None)
    high_sig, _ = librosa.load(os.path.join(inputTargetAudioRoot + relpath), sr=None)

    # Upsample Audio
    low_sig = librosa.resample(low_sig, orig_sr=low_sig_sr, target_sr=inputPredictedAudioSR)
    
    if len(low_sig) < CONFIG.DATA.window_size:
        # padding if the window is longer than the signal
        low_sig = pad(low_sig, CONFIG.DATA.window_size)


    x = frame(low_sig, CONFIG.DATA.window_size, CONFIG.DATA.stride)[:, np.newaxis, :]
    return torch.tensor(x), low_sig, high_sig

def saveAudioAndSpectrogram(lowSignal, predictedSignal, highSignal, spectrogram=True):
    # Get the folder name and file extension
    folderName, fileExtension = os.path.splitext(os.path.basename(inputAudio))
    fileExtension = fileExtension[1:] # remove the . from the file extension
    folderPath = os.path.join(outputFolder, folderName)

    # Create Folders if missing
    if os.path.isdir(folderPath) is not True:
        os.mkdir(folderPath)
    
    # Save the files
    sf.write(os.path.join(folderPath,'low'+'.'+fileExtension) , data=lowSignal,samplerate=inputPredictedAudioSR, format=fileExtension)
    sf.write(os.path.join(folderPath,'predicted'+'.'+fileExtension) , data=predictedSignal,samplerate=inputPredictedAudioSR, format=fileExtension)
    sf.write(os.path.join(folderPath,'target'+'.'+fileExtension) , data=highSignal,samplerate=inputPredictedAudioSR, format=fileExtension)

    # Visualize audio
    fig = plt.figure(figsize=(15, 8))

    # LOW SIGNAL SUBPLOT
    plotSpectrogram(fig, 'Low', lowSignal, inputPredictedAudioSR, 1)

    # PREDICTED SIGNAL SUBPLOT
    plotSpectrogram(fig, 'Predicted', predictedSignal, inputPredictedAudioSR, 2)

    # TARGET SIGNAL SUBPLOT
    plotSpectrogram(fig, 'Target', highSignal, inputPredictedAudioSR, 3)

    plt.subplots_adjust(wspace=0.15)
    plt.savefig(os.path.join(folderPath, 'fig'))
    
    if spectrogram:
        plt.show()

def combineWindows(verticalSignal): # TODO: combine overlapping windows
    horizontalSignal = torch.empty(0)
    for i in verticalSignal.squeeze(1):
        horizontalSignal = torch.cat((horizontalSignal,i))
    return horizontalSignal

def plotSpectrogram(fig, title, data, sr, index):
        # Compute the STFT (Short Time Fourier Transform)
        stft = librosa.stft(data)
        # Convert the STFT matrix to dB scale
        stft_db = librosa.amplitude_to_db(abs(stft))

        # Plot the spectrogram
        fig.add_subplot(1,3,index)
        librosa.display.specshow(stft_db, x_axis='time', y_axis='linear', sr=sr)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()



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
        sf.write(os.path.join(outputFolder,'predicted_p255_001_mic1.flac') , data=horizontalPredSig,samplerate=inputPredictedAudioSR, format='flac')

        # Save audio and show spectrogram
        saveAudioAndSpectrogram(lowSignal,horizontalPredSig,highSignal)
    
if __name__ == '__main__':
    main()
    
