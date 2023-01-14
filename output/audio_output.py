# This file converts low sample rate audio to high sample rate audio using the pretrained model (checkpoints)
import torch
import os
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# Below modules will not be imported without this
import sys
sys.path.append('src/')
from dataset import frame, pad
from model import TUNet
from config import CONFIG

# Global Variables

inputAudioRoot = './data/vctk/8k16k/'
inputAudio = 'p225/p1.flac'
inputCheckpoint = './checkpoints/01-08 PM 10-11-32/Epoch1700_loss2219.pt'

outputFolder = './output/'


def getAudio(path):
    low_sig, low_sig_sr = librosa.load(path, sr=None)

    # Upsample Audio
    low_sig = librosa.resample(low_sig, orig_sr=low_sig_sr, target_sr=16000)
    print(f"lowsig len: {len(low_sig)}")
    if len(low_sig) < CONFIG.DATA.window_size:
        # padding if the window is longer than the signal
        low_sig = pad(low_sig, CONFIG.DATA.window_size)

    x = frame(low_sig, CONFIG.DATA.window_size,
              CONFIG.DATA.stride)[:, np.newaxis, :]
    return torch.tensor(x)


def combineWindows(verticalSignal):  # TODO: combine overlapping windows
    horizontalSignal = torch.empty(0)
    for i in verticalSignal.squeeze(1):
        horizontalSignal = torch.cat((horizontalSignal, i))
    return horizontalSignal


def remove_overlap(signal, window_size, stride):
    # Initialize a list to store the non-overlapping windows
    non_overlap_windows = []
    # calculate the index of windows
    indices = np.arange(0, len(signal) + 1, stride)
    
    
    # Iterate over the windows
    for j,i in enumerate(indices):
        # Append the current window to the list of non-overlapping windows
        if j %2 == 0:
            end =i+stride
            non_overlap_windows.append(signal[i: end])
    if len(indices)%2!=0:
         non_overlap_windows.append(signal[indices[-2]: indices[-1]])
        
    # Concatenate the non-overlapping windows and return the result
    return np.concatenate(non_overlap_windows)

def visualize(hr, lr, recon, path):
    sr = CONFIG.DATA.sr
    window_size = 1024
    window = np.hanning(window_size)

    stft_hr = librosa.core.spectrum.stft(hr, n_fft=window_size, hop_length=512, window=window)
    stft_hr = 2 * np.abs(stft_hr) / np.sum(window)

    stft_lr = librosa.core.spectrum.stft(lr, n_fft=window_size, hop_length=512, window=window)
    stft_lr = 2 * np.abs(stft_lr) / np.sum(window)

    stft_recon = librosa.core.spectrum.stft(recon, n_fft=window_size, hop_length=512, window=window)
    stft_recon = 2 * np.abs(stft_recon) / np.sum(window)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(16, 10))
    ax1.title.set_text('HR signal')
    ax2.title.set_text('LR signal')
    ax3.title.set_text('Reconstructed signal')

    canvas = FigureCanvas(fig)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_hr), ax=ax1, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_lr), ax=ax2, y_axis='linear', x_axis='time', sr=sr)
    p = librosa.display.specshow(librosa.amplitude_to_db(stft_recon), ax=ax3, y_axis='linear', x_axis='time', sr=sr)
    fig.savefig(os.path.join(path, 'spec.png'))
    
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
        horizontalPredSig = combineWindows(predictedSignal).numpy()
        # remove the overlapping windows
        horizontalPredSig = remove_overlap(horizontalPredSig, 8192, 4096)
        print(f"horizontalPredSig len: {len(horizontalPredSig)}")
        # Output the audio
        sf.write(os.path.join(outputFolder, 'predicted_p255_001_mic1.flac'),
                 data=horizontalPredSig, samplerate=16000, format='flac')

        # # Visualize it
        visualize(horizontalPredSig,horizontalPredSig,horizontalPredSig,"./output/")
        # # Compute the STFT (Short Time Fourier Transform)
        # stft = librosa.stft(horizontalPredSig,n_fft=1024, hop_length=512)

        # # Convert the STFT matrix to dB scale
        # stft_db = librosa.amplitude_to_db(abs(stft))

        # # TODO: Add Subplots for comparison
        # # Plot the spectrogram
        # plt.figure(figsize=(10, 5))
        # librosa.display.specshow(stft_db, x_axis='time', y_axis='linear')
        # plt.colorbar()
        # plt.title('Spectrogram')
        # plt.tight_layout()
        # plt.show()


if __name__ == '__main__':
    main()
