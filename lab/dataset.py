from torch.utils.data import Dataset
import torch
import random
from preprocess import preprocess

data_dim = 5


# Define a custom dataset
class StockDataset(Dataset):
    def __init__(self, df, sequence_length, device, percent=1.0):
        self.data = preprocess(df)
        self.sequence_length = sequence_length
        self.device = device
        self.percent = percent
        self.prepare_data()

    def prepare_data(self):
        x_tensors = []
        y_tensors = []
        random_indexes = random.sample(range(self.original_len()), len(self))
        for i in range(len(self)):
            x_tensor, y_tensor = self.getitem(random_indexes[i])
            x_tensors.append(x_tensor)
            y_tensors.append(y_tensor)
        self.x_tensors = x_tensors
        self.y_tensors = y_tensors

    def getitem(self, idx):
        x = self.data[idx : idx + self.sequence_length]
        if x.shape[0] != self.sequence_length:
            return self.getitem(random.randint(0, len(self) - 1))
        y = self.data[idx + self.sequence_length, 1]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        y_tensor = torch.sigmoid(1 - torch.exp(-20 * y_tensor))
        return x_tensor, y_tensor

    def original_len(self):
        return len(self.data) - self.sequence_length - 2

    def __len__(self):
        return int(self.original_len() * self.percent)

    def __getitem__(self, idx):
        return self.x_tensors[idx], self.y_tensors[idx]
