# from time import time
from timeit import default_timer as timer
import multiprocessing as mp
from torch.utils.data import DataLoader
from dataset import CustomDataset
train_data = CustomDataset("train")


def main():
    for num_workers in range(0, mp.cpu_count()):
        train_loader = DataLoader(
            train_data, shuffle=True, collate_fn=CustomDataset.collate_fn, num_workers=num_workers, batch_size=2, pin_memory=True)  # , pin_memory=True

        start = timer()
        for epoch in range(1, 3):
            for i, (X, Y) in enumerate(train_loader):
                X = X.to("cuda")
                Y = Y.to("cuda")
                pass
        end = timer()
        print(
            f"2-Finish with: {end - start:3f} second, num_workers={num_workers}")


if __name__ == '__main__':
    main()
