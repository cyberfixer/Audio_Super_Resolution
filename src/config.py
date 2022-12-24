# This file will have the global variables for the model
# Any setting should be here
import torch
from torch import nn
# This just outline for the config file from the TUNet repo.

# TODO: at the end need to remove the sub classes that are not in use


class CONFIG:

    class TASK:
        pass

    class TRAIN:
        lr = 0.01
        val_split = 0.1

    class MODEL:
        class TRANSFORMER:
            pass

    class DATA:
        dataset = 'vctk'  # dataset to use. Should either be 'vctk' or 'vivos'
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'vctk': {'root': './data/vctk',
                             'train': "./data/train.txt",
                             'trainlow': "./data/train.txt",
                             'test': "./data/test.txt",
                             'testlow': "./data/test.txt"},

                    }
        window_size = 8192  # size of the sliding window
        # stride of the sliding window. Should be divisible to 'mask_chunk' if the task is MSM.
        stride = 4096

    class LOG:
        pass

    class TEST:
        pass
