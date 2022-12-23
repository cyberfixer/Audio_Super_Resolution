# this file will genarate a simple 2 txt files listing and spliting the audio files into a traning set and test set.
# txt files will contain releitve path for every single audio file in the dataset
# there are two files in data folder listing them
import os


def main():
    with open('train.txt', 'w') as f:
        for (root, dirs, files) in os.walk('./vctk', topdown=True):
            for File in files:
                f.write(root[-4:]+'/'+File+'\n')
    f.close()
    print("Done!")


if __name__ == "__main__":
    pass
    #! TODO:need some testing before running it
    # main()
