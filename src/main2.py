# known libraries
from timeit import default_timer as timer
import librosa
import numpy as np
from termcolor import colored as color
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


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
    # ! these list has to be saved & loaded with the model weights
    _trainLoss = np.empty(0)
    _testLoss = np.empty(0)
    _testResulte = np.empty((6, 0))
    epochs = 5
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

            # Zero the gradients
            optimizer.zero_grad()

            # Backward Pass
            trainLoss.backward()

            # Update the model's parameters
            optimizer.step()

        # Train loss for the epoch
        tqdm.write(color(f"Train Loss: {trainLoss:.5f}", "blue"))

        # _trainLoss will contain list of the trainLoss for every epoch
        _trainLoss = np.append(_trainLoss, trainLoss.detach().cpu())
        # tqdm.write(f"shape of _trainLoss: {_trainLoss.shape}")
        # tqdm.write(f"_trainLoss: {_trainLoss})")

        # lr_scheduler contain the optimizer called every epoch
        lr_scheduler.step(trainLoss)

        """Testing"""
        # Set to test mode
        model.eval()
        with torch.inference_mode():
            lsdBatch = np.empty(0)
            lsd_highBatch = np.empty(0)
            sisdrBatch = np.empty(0)
            for batch, (lowSignal, targetSignal) in enumerate(testDataloader):
                # Send to GPU
                lowSignal = lowSignal.to(device)
                targetSignal = targetSignal.to(device)

                # Forward Pass
                predSignal = model(lowSignal)

                # Calculate Loss
                testLoss = lossfn.loss(predSignal, targetSignal)

                # Calculate Metrics
                lsd, lsd_high, sisdr = m.compute_metrics(
                    targetSignal.detach().cpu().numpy(), predSignal.detach().cpu().numpy())

                # Collacting the metrics for the whole batch
                lsdBatch = np.append(lsdBatch, lsd)
                lsd_highBatch = np.append(lsd_highBatch, lsd_high)
                sisdrBatch = np.append(sisdrBatch, sisdr)

            # make the list verticlly
            batchResulte = np.vstack(
                [lsdBatch.mean(0), lsdBatch.std(0),
                 lsd_highBatch.mean(0), lsd_highBatch.std(0), sisdrBatch.mean(0), sisdrBatch.std(0)])
            #
            _testResulte = np.concatenate((_testResulte, batchResulte), axis=1)
            tqdm.write(f"_testResulte.shape:{_testResulte.shape}")
            tqdm.write(f"_testResulte:{_testResulte}")

            # test Loss for every epoch
            tqdm.write(color(f"Test Loss: {testLoss:.5f}", "red"))
            _testLoss = np.append(_testLoss, testLoss.detach().cpu())

        # # TODO: save the model and its variables
        # if epoch % 100:
        #     PATH = f"checkpoints/Epoch{epoch}_loss{int(loss)}.pt"
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss,
        # }, PATH)
    epochlist = list(range(1, epoch+2))
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].plot(epochlist, _testResulte[0])
    ax[0].set_xlabel("Epochs")
    ax[0].set_title("LSD")
    ax[1].plot(epochlist, _testResulte[2])
    ax[1].set_xlabel("Epochs")
    ax[1].set_title("LSD_High")
    ax[2].plot(epochlist, _testResulte[4])
    ax[2].set_xlabel("Epochs")
    ax[2].set_title("SI_SDR")
    plt.show()


if __name__ == "__main__":
    main()
