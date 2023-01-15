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
        batch_size = 2
        lr = 0.0001
        val_split = 0.1
        batch_size = 2  # number of audio files per batch
        epochs = 150  # max training epochs
        """
        # 0 Single process Data Loading
        # workers > 0 Multi process Data Loading
        """
        workers = 2  # number of dataloader workers
        patience = 3  # learning rate scheduler's patience
        factor = 0.5  # learning rate reduction factor
        mse_weight = 10000  # weight of the MSE loss

    class MODEL:
        tfilm = True  # enable/disable TFiLM layers
        n_blocks = 64  # number of blocks of TFiLM layers.
        # bottleneck module. Should either be 'performer', 'lstm' or None
        bottleneck_type = 'performer'
        assert bottleneck_type in ['performer',
                                   'lstm', None], "Invalid bottleneck_type"
        # kernel sizes of each convolution/deconvolution layers
        kernel_sizes = [66, 18, 8]
        strides = [4, 4, 4]  # strides of each convolution/deconvolution layers
        # numbers of filters of each convolution/deconvolution layers
        out_channels = [64, 128, 256]

        # Performer bottleneck config
        class TRANSFORMER:
            dim_head = 32
            depth = 3
            heads = 2

    class DATA:
        dataset = 'vctk'  # dataset to use. Should either be 'vctk' or 'vivos'
        '''
        Dictionary that specifies paths to root directories and train/test text files of each datasets.
        'root' is the path to the dataset and each line of the train.txt/test.txt files should contains the path to an
        audio file from 'root'. 
        '''
        data_dir = {'vctk': {'root': './data/vctk',
                             'train': "./data/train.txt",
                             'trainlow': "./data/trainlow.txt"
                             },

                    }
        window_size = 8192  # size of the sliding window
        # stride of the sliding window. Should be divisible to 'mask_chunk' if the task is MSM.
        stride = 4096
        sr = 16000  # target audio sampling rate

    class LOG:
        pass

    class TEST:
        pass
