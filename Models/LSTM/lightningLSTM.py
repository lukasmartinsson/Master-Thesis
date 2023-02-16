import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

class LSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int):
        super().__init__()

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            batch_first = True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()
        
        out, (h, c) = self.lstm(x)
        h = h[-1]

        return self.fc(h)

class LightningLSTM(pl.LightningModule):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int):
        super().__init__()
        self.model = LSTM(input_size, hidden_size, num_layers)
        self.criterion = nn.MSELoss()

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, output = self(sequences, labels)

        self.log("train_loss", loss, prog_bar=True,logger=True)
        return loss

        
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, output = self(sequences, labels)

        self.log("train_loss", loss, prog_bar=True,logger=True)
        return loss

        
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']

        loss, output = self(sequences, labels)

        self.log("train_loss", loss, prog_bar=True,logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr= 0.0001)