import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch

class DataHandler(Dataset):
    def __init__(self, root: str, mode: str = "train", seq_length: int = 24, batch_size: int = 32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        filepath = os.path.join(root, "38.97_46.5_2019.csv")
        df = pd.read_csv(filepath)
        self.data = df[["Power"]].values

        self.column_size = self.data.shape[1]
        self.row_size = self.data.shape[0] - seq_length - 1
        
        self.max_ = torch.tensor(self.data.max(axis=0)).to(torch.float32)
        self.min_ = torch.tensor(self.data.min(axis=0)).to(torch.float32)

        if mode == "train":
            self.get_sample = self.get_sample_random
            self.limit = 100
        elif mode == "val":
            self.get_sample = self.get_sample_direct
            self.start_idx = - self.batch_size
            self.limit = (
                (self.row_size // self.batch_size) + 
                (self.batch_size - 1 + self.row_size % batch_size) // self.batch_size
            )

    def __call__(self):
        for _ in range(self.limit):
            sample = self.get_sample()
            sample = torch.tensor(sample).to(torch.float32)
            sample = self.scale(sample)
            x = sample[:, :self.seq_length]
            y = sample[:, self.seq_length]
            yield x, y
    
    def get_indexes(self):
        indexes = np.random.choice(self.row_size, self.batch_size)
        return indexes
    
    def get_sample_random(self):
        """Randomly sampling for training purposes."""
        indexes = self.get_indexes()
        sample = np.zeros((self.batch_size, self.seq_length + 1, self.column_size))
        for i, idx in enumerate(indexes):
            sample[i] = self.data[idx: idx + self.seq_length + 1]
        return sample
    
    def get_sample_direct(self):
        """Directly scanning through all samples of the data. Used for generating static validation samples"""
        self.start_idx = self.start_idx + self.batch_size
        if self.start_idx >= (self.row_size - self.batch_size):
            self.start_idx = self.row_size - self.batch_size
        sample = np.zeros((self.batch_size, self.seq_length + 1, self.column_size))
        for i, idx in enumerate(range(self.start_idx, self.start_idx + self.batch_size)):
            sample[i] = self.data[idx: idx + self.seq_length + 1]
            
        return sample
    
    def scale(self, x):
        x = (x - self.min_) / (self.max_ - self.min_)
        return x

if __name__ == "__main__":
    handler = DataHandler("../../../data/", mode="train", seq_length=12, batch_size=32)
    
    handler_generator = handler()
    count = 0
    for x, y in handler_generator:
        pass