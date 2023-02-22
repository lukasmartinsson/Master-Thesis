from torch.utils.data import Dataset, DataLoader
from data_masking import VPPMaskedInputDataset, collate_unsuperv
import pytorch_lightning as pl

class StockDataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences[0])

    def __getitem__(self, idx):
        sequence, label = self.sequences[0][idx], self.sequences[1][idx]
        return dict(sequence = sequence , label = label)

class StockPriceDataModule(pl.LightningDataModule):

    def __init__(self, train_sequence, test_sequence, batch_size:int = 8, num_workers:int = 4):
        super().__init__()
        self.train_sequence = train_sequence
        self.test_sequence = test_sequence
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        self.train_data = StockDataset(self.train_sequence)
        self.test_data = StockDataset(self.test_sequence)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=False, num_workers=self.num_workers)

class MTSTDataModule(pl.LightningDataModule):
    def __init__(self, train_sequence, test_sequence, batch_size:int = 8, num_workers:int = 4):
        super().__init__()
        self.train_sequence = train_sequence
        self.test_sequence = test_sequence
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.training:
            self.train_dataset = VPPMaskedInputDataset(self.train_sequence[0],
                                                       self.train_sequence[1])

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate_unsuperv(x, max_len=self.batch_size),
            drop_last=False)
        print(f"Train iterations: {len(train_dataloader)}")
        return train_dataloader