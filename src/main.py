# this file is the starting point for the model to run everything
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import CONFIG
from model import SSAR, TUNet
from dataset import CustomDataset
from dataset import CustomDataset
from tqdm import tqdm

import librosa
import numpy as np

# Temporarily function
def get_power(x, nfft):
    S = librosa.stft(x, n_fft = nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S

def LSD(x_hr, x_pr): # Log Spectral Distance
    S1 = get_power(x_hr, nfft=2048)
    S2 = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high
# End of Temporarily Functions

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TUNet().to(device)

# Hyperparamters
epochs = 2
batchSize = 4
learningRate = 0.001

def trainStep(trainLoader):  # This is the starting point for training
    # Set to train mode
    model.train()

    # Loss and Optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # Training Loop
    nTotalSteps = len(trainLoader)
    for epoch in range(epochs):
        for i, (lowSignal, targetSignal) in enumerate(tqdm(trainLoader, desc=f'Epoch {epoch+1}')):
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

            if (i+1) % 10 == 0:
                tqdm.write(f'loss {loss.item():.4f}')


def testStep(testLoader):  # This is the starting point testing the model and makeing the voice
    # Set to test mode
    model.eval()

    for i, (lowSignal, targetSignal) in enumerate(tqdm(testLoader)): # ! We only take the first 5 for testing purposes, remove afterwards
        # Send to GPU
        lowSignal = lowSignal.to(device)
        targetSignal = targetSignal.to(device)

        predictedSignal = model(lowSignal)

        lsd, lsd_high = LSD(targetSignal.detach().cpu().numpy(), predictedSignal.detach().cpu().numpy())
        tqdm.write(f' LSD: {np.average(lsd)}, LSD_HIGH: {np.average(lsd_high)}') # must be replaced with something better





def main():
    BATCH_SIZE = 1
    train_data = CustomDataset()
    train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                                  batch_size=BATCH_SIZE,  # how many samples per batch?
                                  shuffle=True,  # shuffle data every epoch?
                                  collate_fn=CustomDataset.collate_fn)
    print(
        f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(train_features_batch.shape, train_labels_batch.shape)

    trainStep(train_dataloader)
    testStep(train_dataloader) # ! must use test_dataloader


if __name__ == '__main__':
    main()
