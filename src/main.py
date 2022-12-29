# this file is the starting point for the model to run everything
from torch.utils.data import DataLoader
from config import CONFIG
from model import SSAR
from dataset import CustomDataset
from dataset import CustomDataset


def train():  # This is the starting point for training
    pass


def test():  # This is the starting point testing the model and makeing the voice
    pass


def main():
    BATCH_SIZE = 32
    train_data = CustomDataset()
    train_dataloader = DataLoader(train_data,  # dataset to turn into iterable
                                  batch_size=1,  # how many samples per batch?
                                  shuffle=True,  # shuffle data every epoch?
                                  collate_fn=CustomDataset.collate_fn)
    print(
        f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(train_features_batch.shape, train_labels_batch.shape)

    for batch, (X, y) in enumerate(train_dataloader):
        print(X.shape)
        print(y.shape)
        break


if __name__ == '__main__':
    main()
