# known libraries
import librosa
import numpy as np
from termcolor import colored as color
from tqdm import tqdm

# classes inside the project
from config import CONFIG
from model import TUNet
from dataset import CustomDataset
from loss import addedLoss as loss
import metrices as m

# torch libraries
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
        shuffle=True,  # shuffle data every epoch?
        collate_fn=CustomDataset.collate_fn,
    )

    # log variables
    _trainLoss = []
    _trainResulte = []
    _testLoss = []
    _testResulte = []
    epochs = 1000
    for epoch in tqdm(range(epochs), desc=f"Total", unit="Epoch"):

        """Training"""
        # Set to train mode
        model.train()

        for batch, (lowSignal, targetSignal) in enumerate(tqdm(trainDataloader, desc="Epoch", unit=" batchs")):
            # Send to GPU
            lowSignal = lowSignal.to(device)
            targetSignal = targetSignal.to(device)

            # Forward Pass
            predSignal = model(lowSignal)

            # Calculate Loss
            trainLoss = lossfn.loss(predSignal, targetSignal)

            # Calculate Metrics
            trainResulte = m.compute_metrics(
                targetSignal.detach().cpu().numpy(), predSignal.detach().cpu().numpy())

            # Backward Pass
            optimizer.zero_grad()
            trainLoss.backward()

            # TODO: running trainloss & trianResulte
            #
            _trainLoss.append(trainLoss)
            _trainResulte.append(trainResulte)

        # lr_scheduler contain the optimizer called every epoch
        lr_scheduler.step(trainLoss)

        """Testing"""
        # Set to test mode
        model.eval()
        with torch.inference_mode():

            for batch, (lowSignal, targetSignal) in enumerate(testDataloader):
                # Send to GPU
                lowSignal = lowSignal.to(device)
                targetSignal = targetSignal.to(device)

                # Forward Pass
                predSignal = model(lowSignal)

                # Calculate Loss
                testLoss = lossfn.loss(predSignal, targetSignal)

                # Calculate Metrics
                testResulte = m.compute_metrics(
                    targetSignal.detach().cpu().numpy(), predSignal.detach().cpu().numpy())

                # TODO: running testloss & testResulte
                #
                _testLoss.append(testLoss)
                _testResulte.append(testResulte)

        # TODO: save the model and its variables
        if epoch % 100:
            pass


if __name__ == "__main__":
    main()
