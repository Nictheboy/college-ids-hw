from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random

data_dim = 6


# Define a custom dataset
class StockDataset(Dataset):
    def __init__(self, df, sequence_length, device, percent=1.0):
        self.data, self.scaler = self.preprocess_data(df)
        self.sequence_length = sequence_length
        self.device = device
        self.percent = percent
        self.prepare_data()

    # Data preparation
    def preprocess_data(self, df):
        if "Adj Close" in df.columns:
            del df["Adj Close"]

        # Normalize date
        df["Date Time"] = pd.to_datetime(df["Date Time"])
        df = df.sort_values("Date Time")
        date_begin = pd.to_datetime("2010-01-04 00:00:00")
        date_end = pd.to_datetime("2022-12-30 00:00:00")
        df["Date Time"] = (df["Date Time"] - date_begin) / (date_end - date_begin)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        return scaled_data, scaler

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
            print("Warning: sequence length not enough")
            return self.getitem(random.randint(0, len(self) - 1))
        y = (
            1
            if self.data[idx + self.sequence_length + 1, 1]
            > self.data[idx + self.sequence_length, 1]
            else 0
        )
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        return x_tensor, y_tensor

    def original_len(self):
        return len(self.data) - self.sequence_length - 1

    def __len__(self):
        return int(self.original_len() * self.percent)

    def __getitem__(self, idx):
        return self.x_tensors[idx], self.y_tensors[idx]
