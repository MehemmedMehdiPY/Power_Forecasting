"""
LSTM architecture is based on the paper 
        Alex G., et al "Generating Sequences with Recurrent Neural Networks"
        available at https://arxiv.org/abs/1308.0850
"""

import torch
from torch import nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.fc_ii = nn.Linear(input_size, hidden_size)
        self.fc_hi = nn.Linear(hidden_size, hidden_size)

        self.fc_if = nn.Linear(input_size, hidden_size)
        self.fc_hf = nn.Linear(hidden_size, hidden_size)
        self.fc_cf = nn.Linear(hidden_size, hidden_size)

        self.fc_ic = nn.Linear(input_size, hidden_size)
        self.fc_hc = nn.Linear(hidden_size, hidden_size)

        self.fc_io = nn.Linear(input_size, hidden_size)
        self.fc_ho = nn.Linear(hidden_size, hidden_size)
        self.fc_co = nn.Linear(hidden_size, hidden_size)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, previous_gate):
        hidden_gate, cell_gate = previous_gate
        input_gate = self.sigmoid(
            self.fc_ii(x) + self.fc_hi(hidden_gate)
            )
        forget_gate = self.sigmoid(
            self.fc_if(x) + self.fc_hf(hidden_gate) + self.fc_cf(cell_gate)
        )
        cell_gate = (
            forget_gate * cell_gate + 
            input_gate * self.tanh(self.fc_ic(x) + self.fc_hc(hidden_gate)) 
        )
        output_gate = self.sigmoid(self.fc_io(x) + self.fc_ho(hidden_gate) + self.fc_co(cell_gate))        
        hidden_gate = output_gate * self.tanh(cell_gate)
        return output_gate, (hidden_gate, cell_gate)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.fc_final = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_size, _ = x.size()
        hidden_gate = torch.zeros(batch_size, self.hidden_size)
        cell_gate = torch.zeros(batch_size, self.hidden_size)
        for idx in range(seq_size):
            x_input = x[:, idx, :]
            output_gate, (hidden_gate, cell_gate) = self.cell(x_input, (hidden_gate, cell_gate))
        output_gate = self.fc_final(output_gate)    
        return output_gate, (hidden_gate, cell_gate)

if __name__ == "__main__":
    
    model = LSTM(input_size=3, hidden_size=32)
    x = torch.randn(size=(1, 5, 3))
    print(x.shape)
    out, (h, c) = model(x)
    print(out.shape, h.shape, c.shape)