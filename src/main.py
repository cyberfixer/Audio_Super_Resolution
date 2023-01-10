# known libraries
from timeit import default_timer as timer
import librosa
import os
import numpy as np
from termcolor import colored as color
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


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


def main():
    # device agnastic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # creating the model and sending it to the device
    model = TUNet().to(device)

    # hyperParameters
    BATCH_SIZE = CONFIG.TRAIN.batch_size
    LR = CONFIG.TRAIN.lr
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
    # ! these lists has to be saved & loaded with the model weights
    _trainLoss = np.empty(0)
    _testLoss = np.empty(0)
    _testResulte = np.empty((6, 0))

    # this variable will determen that is new train or will load a model
    newTrain = True
    if newTrain == True:
        now = datetime.now()  # current date and time
        # folder name for the checkpoints
        # strftime convert time to string
        folder = now.strftime("%m-%d %p %I-%M-%S")
        print(folder)
        try:
            os.makedirs(f"checkpoints/{folder}")
        except FileExistsError:
            pass
    else:
        """this part will contain torch.load and will load all the variables needed"""
        pass

    epochs = 2500
    for epoch in tqdm(range(epochs), desc=f"Total", unit="Epoch", dynamic_ncols=True):

        """Training"""
        # Set to train mode
        model.train()
        trainLoss = 0
        num_samples = 0
        for batch, (lowSignal, targetSignal) in enumerate(tqdm(trainDataloader, desc="Epoch", unit=" batchs", dynamic_ncols=True)):
            # Send to GPU
            lowSignal = lowSignal.to(device)
            targetSignal = targetSignal.to(device)

            # Forward Pass
            predSignal = model(lowSignal)

            # Calculate Loss
            lossBatch = lossfn.loss(predSignal, targetSignal)
            # the will add up the losses
            # lowSignal.size(0) is the batch size
            trainLoss += lossBatch * lowSignal.size(0)
            num_samples += lowSignal.size(0)
            # Zero the gradients
            optimizer.zero_grad()

            # Backward Pass
            lossBatch.backward()

            # Update the model's parameters
            optimizer.step()

        # trainloss avrage
        trainLoss /= num_samples
        # _trainLoss will contain list of the trainLoss for every epoch
        _trainLoss = np.append(_trainLoss, trainLoss.detach().cpu())

        """Testing"""
        # Set to test mode
        model.eval()
        with torch.inference_mode():
            lsdBatch = np.empty(0)
            lsd_highBatch = np.empty(0)
            sisdrBatch = np.empty(0)
            testLoss = 0
            num_samples = 0
            for batch, (lowSignal, targetSignal) in enumerate(testDataloader):
                # Send to GPU
                lowSignal = lowSignal.to(device)
                targetSignal = targetSignal.to(device)

                # Forward Pass
                predSignal = model(lowSignal)

                # Calculate Loss
                lossBatch = lossfn.loss(predSignal, targetSignal)

                # the will add up the losses
                # lowSignal.size(0) is the batch size
                testLoss += lossBatch * lowSignal.size(0)
                num_samples += lowSignal.size(0)

                # Calculate Metrics
                lsd, lsd_high, sisdr = m.compute_metrics(
                    targetSignal.detach().cpu().numpy(), predSignal.detach().cpu().numpy())

                # Collacting the metrics for the whole batch
                lsdBatch = np.append(lsdBatch, lsd)
                lsd_highBatch = np.append(lsd_highBatch, lsd_high)
                sisdrBatch = np.append(sisdrBatch, sisdr)

            # lr_scheduler contain the optimizer called every epoch
            lr_scheduler.step(testLoss)
            # Compute the average loss
            testLoss /= num_samples
            # _testLoss will contain list of the testLoss for every epoch
            _testLoss = np.append(_testLoss, testLoss.detach().cpu())

            # make the list verticlly
            batchResulte = np.vstack(
                [lsdBatch.mean(0), lsdBatch.std(0),
                 lsd_highBatch.mean(0), lsd_highBatch.std(0), sisdrBatch.mean(0), sisdrBatch.std(0)])
            # adding the batchResulte to _testResulte
            _testResulte = np.concatenate((_testResulte, batchResulte), axis=1)

        # PATH of the checkpoint
        PATH = f"./checkpoints/{folder}/Epoch{epoch}_loss{int(_testLoss[-1])}.pt"
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                '_trainloss': _trainLoss,
                '_testloss': _testLoss,
                '_testResulte': _testResulte
            }, PATH)

        # this will make log.txt file
        with open(f"./checkpoints/{folder}/log.txt", "a") as f:
            f.write(f"----------------{epoch}----------------\n")
            f.write(f"Train Loss: {trainLoss:.5f}\n")
            f.write(f"Test Loss: {testLoss:.5f}\n")
            f.write(f"LSD Mean: {batchResulte[0]}\n")
            f.write(f"LSD STD: {batchResulte[1]}\n")
            f.write(f"LSD-High Mean: {batchResulte[2]}\n")
            f.write(f"LSD-High STD: {batchResulte[3]}\n")
            f.write(f"SI-SDR Mean: {batchResulte[4]}\n")
            f.write(f"SI-SDR STD: {batchResulte[5]}\n")


if __name__ == "__main__":
    main()
