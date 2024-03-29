# this file should have every code for handling the dataset
# we have to make it in a class

import os
import random
import soundfile as sf
import librosa
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import CONFIG
from torch.utils.data import DataLoader


def pad(sig, length):
    # padding with 0.1 at the end of the signal
    if len(sig) < length:
        pad = length - len(sig)
        sig = np.hstack((sig, np.zeros(pad) + 0.1))
    else:
        start = random.randint(0, len(sig) - length)
        sig = sig[start:start + length]
    return sig


def frame(a, w, s, copy=True):
    # stacking the windows vertically
    # retrun (window_number, amplitude samples)
    if len(a) < w:
        return np.expand_dims(np.hstack((a, np.zeros(w - len(a)))), 0)

    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::s]

    if copy:
        return view.copy()
    else:
        return view


class CustomDataset(Dataset):
    # todo we may need to move the config variables to the head of __init__
    def __init__(self, mode='train'):
        data_dir = CONFIG.DATA.data_dir
        name = CONFIG.DATA.dataset
        self.target_root = data_dir[name]['root']
        # path of the txt files
        txt_list_low = data_dir[name]['trainlow']
        txt_list_target = data_dir[name]['train']
        # open the txt files
        self.data_list_low = self.load_txt(txt_list_low)
        self.data_list_target = self.load_txt(txt_list_target)
        # split the data to train and test
        if mode == 'train':  # (X,y) -> (low , target)
            self.data_list_low, _ = train_test_split(
                self.data_list_low, test_size=CONFIG.TRAIN.val_split, random_state=0)
            self.data_list_target, _ = train_test_split(
                self.data_list_target, test_size=CONFIG.TRAIN.val_split, random_state=0)
        elif mode == 'test':
            _, self.data_list_low = train_test_split(
                self.data_list_low, test_size=CONFIG.TRAIN.val_split, random_state=0)
            _, self.data_list_target = train_test_split(
                self.data_list_target, test_size=CONFIG.TRAIN.val_split, random_state=0)

        self.mode = mode
        self.window = CONFIG.DATA.window_size
        self.stride = CONFIG.DATA.stride

    def __len__(self):
        return len(self.data_list_target)  # get the length of the data set

    @staticmethod  # staticmethod so we can call it without an object
    def collate_fn(batch):  # make the batch one tensor
        data = torch.cat([item[0] for item in batch], dim=0).float()
        target = torch.cat([item[1] for item in batch], dim=0).float()
        return [data, target]

    def load_txt(self, txt_list):
        """it opens the path in txt_list and returen the content"""
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(target)
        return target

    def __getitem__(self, index):

        low_sig, _ = librosa.load(self.data_list_low[index], sr=None)
        target, _ = librosa.load(self.data_list_target[index], sr=None)


        if len(low_sig) < self.window:
            # padding if the window is longer than the signal
            low_sig = pad(low_sig, self.window)
        if len(target) < self.window:
            # padding if the window is longer than the signal
            target_sig = pad(target_sig, self.window)

        if len(target) != len(low_sig):
            low_sig = pad(low_sig, len(target))
        # stacks the windows virticlay X=(window_number, amplitude samples)
        # there was in this sliceing numpy.newaxis

        X = frame(low_sig, self.window, self.stride)[:, np.newaxis, :]
        if self.mode == 'test1':
            return X, target, low_sig

        # stacks the windows virticlay y=(window_number, amplitude samples)
        # there was in this sliceing numpy.newaxis #,
        y = frame(target, self.window, self.stride)[:, np.newaxis, :]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def main():  # ! this main is just for testing the CustomDataset class
    DATaset = CustomDataset()
    print(f" there are {len(DATaset)} audio files")
    # change the index # it will call the __getitem__
    low_sig, high_sig = DATaset[1]

    print(f"shape of low_sig:{low_sig.shape}")
    print(f"shape of high_sig:{high_sig.shape}")
    # data_loader = DataLoader(DATaset, shuffle=False,
    #                          batch_size=16, collate_fn=CustomDataset.collate_fn)
    # for batch, (X, Y) in enumerate(data_loader):
    #     print(f"shape X: {X.shape}")
    # print(d)
    # this loop append the windows to each other
    sigh = torch.zeros(1)
    for i in high_sig[:, 0]:
        sigh = torch.cat((sigh, i))

    sigl = torch.zeros(1)
    for i in low_sig[:, 0]:
        sigl = torch.cat((sigl, i))
    # print(sig.shape)
    print(f"low shape: {sigl.shape}")
    print(f"high shape: {sigh.shape}")
    sf.write("high_sig.flac",  sigh.detach().numpy(), samplerate=48000)
    sf.write("low_sig.flac",  sigl.detach().numpy(), samplerate=8000)


if __name__ == '__main__':
    main()
"""
if the sound is cliping change the CONFIG.DATA.STRIDE to 8192 to match CONFIG.DATA.window_size
"""
