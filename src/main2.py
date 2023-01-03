#
import librosa
import numpy as np
from termcolor import colored as color
from tqdm import tqdm

# classes inside the project
from config import CONFIG
from model import TUNet
from dataset import CustomDataset
from loss import MRSTFTLossDDP
from loss import loss

# torch libaries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# device agnastic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# creating the model and sending it to the device
model = TUNet().to(device)


def main():
    # hyperParameters
    BATCH_SIZE = 2
    LR = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        patience=CONFIG.TRAIN.patience,
        factor=CONFIG.TRAIN.factor,
        verbose=True,
    )
    # loss class contain MSE & MRSTFTlossDDP * 10000
    lossfn = loss()

    # importing data
    train_data = CustomDataset("train")
    test_data = CustomDataset("test")
    trainDataloader = DataLoader(
        train_data,  # dataset to turn into iterable
        batch_size=BATCH_SIZE,  # how many samples per batch?
        shuffle=True,  # shuffle data every epoch?
        collate_fn=CustomDataset.collate_fn,
    )
    testDataloader = DataLoader(
        test_data,  # dataset to turn into iterable
        batch_size=BATCH_SIZE,  # how many samples per batch?
        shuffle=False,  # shuffle data every epoch?
        collate_fn=CustomDataset.collate_fn,
    )

    epochs = 1000
    for epoch in tqdm(range(epochs), desc=f"Total", unit=" Epochs"):

        """Training"""
        # Set to train mode
        model.train()

        for batch, (lowSignal, targetSignal) in enumerate(tqdm(trainDataloader, desc="Epoch", unit=" batchs")):
            # Send to GPU
            lowSignal = lowSignal.to(device)
            targetSignal = targetSignal.to(device)

            # Forward Pass
            predictedSignal = model(lowSignal)

            # Calculate Loss
            Trainloss = lossfn(predictedSignal, targetSignal)

            # Calculate Metrics
            # TODO: Calculate Metrics

            # Backward Pass
            optimizer.zero_grad()
            Trainloss.backward()

            # lr_scheduler contain the optimizer
            lr_scheduler.step(Trainloss)

        """Testing"""
        # Set to test mode
        model.eval()
        with torch.inference_mode():

            for batch, (lowSignal, targetSignal) in enumerate(testDataloader):
                # Send to GPU
                lowSignal = lowSignal.to(device)
                targetSignal = targetSignal.to(device)

                # Forward Pass
                predictedSignal = model(lowSignal)

                # Calculate Loss
                Testloss = lossfn(predictedSignal, targetSignal)

                # Calculate Metrics
                # TODO: Calculate Metrics


if __name__ == "__main__":
    main()
