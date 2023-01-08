import librosa
import soundfile as sf
from glob import glob  # get full path of all files
from tqdm import tqdm  # progress bar
import os
import threading

inputFilesPath = './data/vctk/48k'
outputFilesPath = './data/vctk/8k'
targetsr = 8000

workers = 8
FoldersInProcess = []



def innerFolders(folders):
    lock = threading.Lock()
    
    for folderPath in folders:  # All folders
        targetFolderPath = os.path.join(outputFilesPath, os.path.basename(folderPath))

        # Is it already being worked on by another thread?
        if targetFolderPath in FoldersInProcess:
            continue

        # Add the folder in foldersInProgress
        lock.acquire()
        FoldersInProcess.append(targetFolderPath)
        lock.release()

        # Create Folders if missing
        if os.path.isdir(targetFolderPath) is not True:
            os.mkdir(targetFolderPath)

        # All files in each folder
        g = glob(folderPath + '/*')
        for filePath in tqdm(g, desc=os.path.basename(folderPath), leave=False):

            # Path Calculculations
            fileName = os.path.basename(filePath)
            targetFilePath = os.path.join(targetFolderPath, fileName)
            targetFileExtension = os.path.splitext(fileName)[1][1:]

            # Does the file exist?
            if os.path.exists(targetFilePath):
                continue

            # Downsampling the audio
            signal, samplerate = librosa.load(filePath, sr=None)
            signalLow = librosa.resample(
                signal, orig_sr=samplerate, target_sr=targetsr)

            # Saving the audio file.
            sf.write(targetFilePath, data=signalLow,
                    samplerate=targetsr, format=targetFileExtension)

        # Remove the folder in foldersInProgress
        lock.acquire()
        FoldersInProcess.remove(targetFolderPath)
        lock.release()

def main():
    # Browse all Input Folders
    folders = glob(inputFilesPath + '/*')

    # Create Outputs folder if missing
    if os.path.isdir(outputFilesPath) is not True:
        os.mkdir(outputFilesPath)

    threads = [threading.Thread(target=innerFolders, args=[folders]) for i in range(workers)]
    
    # Start threads
    for i in range(workers):
        threads[i].start()
        
    # Wait for other threads to finish
    for i in range(workers):
        threads[i].join()
        


if __name__ == '__main__':

    main()
