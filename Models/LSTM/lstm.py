import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) #Fully connected layer

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0))
        out, (hn, cn) = self.lstm(out, (h_0, c_0))
        out, (hn, cn) = self.lstm(out, (h_0, c_0))
        # Select last output
        out = out[:, -1, :]
        # Pass through fully connected layer
        out = self.fc(out)
        return out
