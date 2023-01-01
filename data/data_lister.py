# this file will genarate a simple 2 txt files listing and spliting the audio files into a traning set and test set.
# txt files will contain releitve path for every single audio file in the dataset
# there are two files in data folder listing them
import os
from tqdm import tqdm


def main():
    """
    the inteded behavier is "folder/filename" 
    """
    with open('data/train.txt', 'w+') as f:
        for (root, dirs, files) in tqdm(os.walk('./data/vctk/', topdown=True)):
            for fileName in files:
                folderName = os.path.basename(root)
                filePath = folderName + '/' + fileName
                f.write(filePath+"\n")
    f.close()
    print("Done!")


if __name__ == "__main__":
    main()
