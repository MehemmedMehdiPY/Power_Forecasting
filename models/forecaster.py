from .lstm import LSTM
from torch import nn

class Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x
    

if __name__ == "__main__":
    import torch
    x = torch.randn(1, 5, 3)
    model = Forecaster(input_size=3)
    out = model(x)
    print(out.shape, out)