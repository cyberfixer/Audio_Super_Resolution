
from dtw import dtw
import numpy as np

import torch
from tqdm.auto import tqdm
from config import CONFIG
from model import TUNet
from dataset import CustomDataset
from loss import addedLoss as loss
import metrices as m
from torch.utils.data import DataLoader

# device agnastic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# creating the model and sending it to the device
model = TUNet().to(device)

BATCH_SIZE = CONFIG.TRAIN.batch_size

# loss class contain MSE & MRSTFTlossDDP * 10000
lossfn = loss()
test_data = CustomDataset("All")
testDataloader = DataLoader(
    test_data,  # dataset to turn into iterable
    batch_size=BATCH_SIZE,  # how many samples per batch?
    collate_fn=CustomDataset.collate_fn,
    num_workers=0,
    pin_memory=True,
)
PATH = "./checkpoints/01-17 AM 11-47-52/Epoch140_loss1301.pt"
checkpoint = torch.load(PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
testLoss = 0
num_samples = 0
with torch.inference_mode():
    lsdBatch = np.empty(0)
    lsd_highBatch = np.empty(0)
    sisdrBatch = np.empty(0)
    for batch, (lowSignal, targetSignal) in enumerate(tqdm(testDataloader, desc="Test", unit=" batchs", leave=False, dynamic_ncols=True)):
        # Send to GPU
        lowSignal = lowSignal.to(device)
        targetSignal = targetSignal.to(device)

        # Forward Pass
        predSignal = model(lowSignal)

        # Calculate Loss
        lossBatch = lossfn.loss(predSignal, targetSignal)
        # the will add up the losses
        # lowSignal.size(0) is the batch size
        testLoss += lossBatch.detach().cpu() * lowSignal.size(0)
        num_samples += lowSignal.size(0)
        
        # Calculate Metrics
        lsd, lsd_high, sisdr = m.compute_metrics(
            targetSignal.detach().cpu().numpy(), predSignal.detach().cpu().numpy())

        # Collacting the metrics for the whole batch
        lsdBatch = np.append(lsdBatch, lsd)
        lsd_highBatch = np.append(lsd_highBatch, lsd_high)
        sisdrBatch = np.append(sisdrBatch, sisdr)
        
    # Compute the average loss
    testLoss /= num_samples
    
    # make the list verticlly
    batchResulte = np.vstack(
        [lsdBatch.mean(0), lsdBatch.std(0),
         lsd_highBatch.mean(0), lsd_highBatch.std(0), 
         sisdrBatch.mean(0), sisdrBatch.std(0)])
    
    # log the metrices to text file 
    with open(f"./output/FD 140epochs/metrices.txt", "a") as f:
            f.write(f"Test Loss: {testLoss:.5f}\n")
            f.write(f"LSD Mean: {batchResulte[0]}\n")
            f.write(f"LSD STD: {batchResulte[1]}\n")
            f.write(f"LSD-High Mean: {batchResulte[2]}\n")
            f.write(f"LSD-High STD: {batchResulte[3]}\n")
            f.write(f"SI-SDR Mean: {batchResulte[4]}\n")
            f.write(f"SI-SDR STD: {batchResulte[5]}\n")