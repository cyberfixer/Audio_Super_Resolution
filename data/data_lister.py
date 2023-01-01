# this file will genarate a simple 2 txt files listing and spliting the audio files into a traning set and test set.
# txt files will contain releitve path for every single audio file in the dataset
# there are two files in data folder listing them
import os
from tqdm import tqdm


def listerTrain():
    with open('data/train.txt', 'w+') as f:
        for (root, dirs, files) in tqdm(os.walk('./data/vctk/48k', topdown=True)):
            for fileName in files:
                parentFolderName = os.path.basename(os.path.split(root)[0])
                folderName = os.path.basename(root)
                filePath = parentFolderName + '/' + folderName + '/' + fileName
                f.write(filePath+"\n")
    f.close()
    print("Done!")


def listerTrainlow(samplingRateFolderName):
    with open('data/trainlow.txt', 'w+') as f:
        for (root, dirs, files) in tqdm(os.walk('./data/vctk/'+samplingRateFolderName, topdown=True)):
            for fileName in files:
                parentFolderName = os.path.basename(os.path.split(root)[0])
                folderName = os.path.basename(root)
                filePath = parentFolderName + '/' + folderName + '/' + fileName
                f.write(filePath+"\n")
    f.close()
    print("Done!")


def main():

    samplingRateFolderName = "8k"
    listerTrain()
    listerTrainlow(samplingRateFolderName)


if __name__ == "__main__":
    main()
