import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch

class DataHandler(Dataset):
    def __init__(self, root: str, seq_length: int = 24, batch_size: int = 32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        filepath = os.path.join(root, "38.97_46.5_2019.csv")
        df = pd.read_csv(filepath)
        self.data = df[["Power"]].values

        self.column_size = self.data.shape[1]
        self.row_size = self.data.shape[0] - seq_length - 1

        self.max_ = torch.tensor(self.data.max(axis=0)).to(torch.float32)
        self.min_ = torch.tensor(self.data.min(axis=0)).to(torch.float32)

    def __call__(self):
        sample = self.get_sample()
        sample = torch.tensor(sample).to(torch.float32)
        sample = (sample - self.min_) / (self.max_ - self.min_)
        x = sample[:, :self.seq_length]
        y = sample[:, self.seq_length]
        return x, y
    
    def get_indexes(self):
        indexes = np.random.choice(self.row_size, self.batch_size)
        return indexes
    
    def get_sample(self):
        indexes = self.get_indexes()
        sample = np.zeros((self.batch_size, self.seq_length + 1, self.column_size))
        for i, idx in enumerate(indexes):
            sample[i] = self.data[idx: idx + self.seq_length + 1]
        return sample


if __name__ == "__main__":

    handler = DataHandler("../../../data/", seq_length=24, batch_size=32)
    print(handler.min_, handler.max_)
    x, y = handler()
    print(x.shape, y.shape)
    