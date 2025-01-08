from .LSTM import LSTM
import torch
from torch import nn

class Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, 64)
    def forward(self, x):
        return x