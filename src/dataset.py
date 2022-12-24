# this file should have every code for handling the dataset
# we have to make it in a class

import os
import random

import librosa
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import CONFIG


def pad(sig, length):
    if len(sig) < length:
        pad = length - len(sig)
        sig = np.hstack((sig, np.zeros(pad) + 0.1))
    else:
        start = random.randint(0, len(sig) - length)
        sig = sig[start:start + length]
    return sig


def frame(a, w, s, copy=True):
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

    # def __init__(self, mode='train'):
    #     data_dir = CONFIG.DATA.data_dir
    #     name = CONFIG.DATA.dataset
    #     self.target_root = data_dir[name]['root']
    #     if mode == 'test':
    #         txt_list = data_dir[name]['test']
    #     else:
    #         txt_list = data_dir[name]['train']
    #     self.data_list = self.load_txt(txt_list)
    #     if mode == 'train':
    #         self.data_list, _ = train_test_split(
    #             self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)
    #     elif mode == 'val':
    #         _, self.data_list = train_test_split(
    #             self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)
    #     self.mode = mode
    #     self.sr = CONFIG.DATA.sr
    #     self.down_rate = CONFIG.DATA.ratio
    #     self.window = CONFIG.DATA.window_size
    #     self.stride = CONFIG.DATA.stride
    #     self.task = CONFIG.TASK.task
    #     self.downsampling = CONFIG.TASK.downsampling

    def __init__(self, mode='train'):
        data_dir = CONFIG.DATA.data_dir
        name = CONFIG.DATA.dataset
        self.target_root = data_dir[name]['root']
        if mode == 'train':
            txt_list_low = data_dir[name]['trainlow']
            txt_list_target = data_dir[name]['train']
        else:
            txt_list_low = data_dir[name]['testlow']
            txt_list_target = data_dir[name]['test']

        self.data_list_low = self.load_txt(txt_list_low)
        self.data_list_target = self.load_txt(txt_list_target)

        if mode == 'train':
            self.data_list_low, _ = train_test_split(
                self.data_list_low, test_size=CONFIG.TRAIN.val_split, random_state=0)
            self.data_list_low, _ = train_test_split(
                self.data_list_low, test_size=CONFIG.TRAIN.val_split, random_state=0)
        elif mode == 'val':
            _, self.data_list_target = train_test_split(
                self.data_list_target, test_size=CONFIG.TRAIN.val_split, random_state=0)
            _, self.data_list_target = train_test_split(
                self.data_list_target, test_size=CONFIG.TRAIN.val_split, random_state=0)

        self.mode = mode
        self.window = CONFIG.DATA.window_size
        self.stride = CONFIG.DATA.stride

    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(set(target))
        target.sort()
        return target

    def __getitem__(self, index):
        low_sig, sr = librosa.load(self.data_list_low[index], sr=None)
        target, sr = librosa.load(self.data_list_target[index], sr=None)

        if len(low_sig) < self.window:
            # padding if the window is longer than the signal
            low_sig = pad(low_sig, self.window)
        if len(target) < self.window:
            # padding if the window is longer than the signal
            target_sig = pad(target_sig, self.window)

        X = frame(low_sig, self.window, self.stride)[:, np.newaxis, :]
        # if self.mode == 'test':
        #     return X, target, low_sig

        y = frame(target, self.window, self.stride)[:, np.newaxis, :]
        return torch.tensor(X), torch.tensor(y)


def main():
    DATaset = CustomDataset()
    print(f" there are {len(DATaset)}")


if __name__ == '__main__':
    main()
