# this file is the starting point for the model to run everything

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG
from model import SSAR, TUNet
from dataset import CustomDataset
from dataset import CustomDataset


import librosa
import numpy as np
from termcolor import colored as color
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# Temporarily function


def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


def LSD(x_hr, x_pr):  # Log Spectral Distance
    S1 = get_power(x_hr, nfft=2048)
    S2 = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(
        np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high
# End of Temporarily Functions


def trainStep(model: torch.nn.Module,
              trainLoader: torch.utils.data.DataLoader,
              lossFunction: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device = device):  # This is the starting point for training
    # Set to train mode
    model.train()

    # Training Loop
    for batch, (lowSignal, targetSignal) in enumerate(trainLoader):
        # Send to GPU
        lowSignal = lowSignal.to(device)
        targetSignal = targetSignal.to(device)

        # Forward Pass
        predictedSignal = model(lowSignal)
        loss = lossFunction(predictedSignal, targetSignal)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(color(f"Train Loss: {loss:.5f}", "blue"))


def testStep(model: torch.nn.Module,
             testLoader: torch.utils.data.DataLoader,
             lossFunction: torch.nn.Module,
             device: torch.device = device):  # This is the starting point testing the model and makeing the voice
    # Set to test mode
    model.eval()
    results = []
    resultsLSD, resultsLSDHigh = np.empty(0), np.empty(0)
    # ! We only take the first 5 for testing purposes, remove afterwards
    with torch.inference_mode():
        for batch, (lowSignal, targetSignal) in enumerate(testLoader):
            # Send to GPU
            lowSignal = lowSignal.to(device)
            targetSignal = targetSignal.to(device)

            predictedSignal = model(lowSignal)
            loss = lossFunction(predictedSignal, targetSignal)

            lsd, LSDHigh = LSD(targetSignal.detach().cpu().numpy(),
                               predictedSignal.detach().cpu().numpy())

            resultsLSD = np.append(resultsLSD, lsd)
            resultsLSDHigh = np.append(resultsLSDHigh, LSDHigh)

    results = [resultsLSD.mean(0), resultsLSD.std(
        0), resultsLSDHigh.mean(0), resultsLSDHigh.std(0)]
    print(color(f"Test Loss: {loss:.5f}", "red"))
    print(f"LSD Mean: {results[0]:.3f}")
    print(f"LSD STD: {results[1]:.3f}")
    print(f"LSD-HF Mean: {results[2]:.3f}")
    print(f"LSD-HF STD: {results[3]:.3f}")


def main():
    model = TUNet().to(device)
    # heperparamters
    BATCH_SIZE = 1
    learningRate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    lossFunction = nn.CrossEntropyLoss()

    train_data = CustomDataset()
    train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                                  batch_size=BATCH_SIZE,  # how many samples per batch?
                                  shuffle=True,  # shuffle data every epoch?
                                  collate_fn=CustomDataset.collate_fn)

    print(
        f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    # train_features_batch, train_labels_batch = next(iter(train_dataloader))
    # print(train_features_batch.shape, train_labels_batch.shape)
    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f" Epoch: {epoch}\n---------")
        trainStep(model, train_dataloader, lossFunction, optimizer)
        # ! must use test_dataloader
        testStep(model, train_dataloader, lossFunction)


if __name__ == '__main__':
    main()
