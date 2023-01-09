import librosa
import soundfile as sf
from glob import glob  # get full path of all files
from tqdm import tqdm  # progress bar
import os

inputFilesPath = './data/vctk/48k'
outputFilesPath = './data/vctk/8k resampled 16k'

targetsr = 16000
datasr = 8000


def main():
    folders = glob(inputFilesPath + '/*')
    for folderPath in tqdm(folders, desc='Total',  dynamic_ncols=True):  # All folders
        # All files in each folder
        for filePath in tqdm(glob(folderPath + '/*'), desc=os.path.basename(folderPath), leave=False, dynamic_ncols=True):

            # Path Calculculations
            fileName = os.path.basename(filePath)
            targetFolderPath = os.path.join(
                outputFilesPath, os.path.basename(folderPath))
            targetFilePath = os.path.join(targetFolderPath, fileName)
            targetFileExtension = os.path.splitext(fileName)[1][1:]

            # Does the file exist?
            if os.path.exists(targetFilePath):
                continue

            # Create Folders if missing
            if os.path.isdir(outputFilesPath) is not True:
                os.mkdir(outputFilesPath)
            if os.path.isdir(targetFolderPath) is not True:
                os.mkdir(targetFolderPath)

            # Downsampling the audio
            signal, samplerate = librosa.load(filePath, sr=None)
            # If the datasr is not the same as targetsr
            if datasr < targetsr:
                # Downsample the data
                signal = librosa.resample(
                    signal, orig_sr=samplerate, target_sr=datasr)
                samplerate = datasr
            # After that it will up sample it to targetsr
            signalLow = librosa.resample(
                signal, orig_sr=samplerate, target_sr=targetsr)

            # Saving the audio file.
            sf.write(targetFilePath, data=signalLow,
                     samplerate=targetsr, format=targetFileExtension)


if __name__ == '__main__':
    main()
