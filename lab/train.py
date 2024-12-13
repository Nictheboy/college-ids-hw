import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


data_dim = 6


# Define a custom dataset
class StockDataset(Dataset):
    def __init__(self, filename, sequence_length, device):
        self.data, self.scaler = StockDataset.preprocess_data(pd.read_csv(filename))
        self.sequence_length = sequence_length
        self.device = device

    # Data preparation
    def preprocess_data(df):
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

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.sequence_length]
        y = (
            1
            if self.data[idx + self.sequence_length + 1, -2]
            > self.data[idx + self.sequence_length, -2]
            else 0
        )
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        return x_tensor, y_tensor


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=1,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x) + self.positional_encoding[:, : x.size(1), :]
        x = self.transformer(x, x)
        x = x[:, -1, :]  # Use the last time step's output
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


# Hyperparameters
sequence_length = 30
batch_size = 64
d_model = 1024
nhead = 32
num_layers = 32
dim_feedforward = 128
epochs = 10
lr = 0.001

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def create_model(filename: str):
    model = TransformerModel(
        input_dim=data_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    torch.save(model.state_dict(), filename)


def train_model(filename: str):
    # Create dataset and dataloaders
    dataset = StockDataset("data/converted/000001.SZ.csv", sequence_length, device=device)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = TransformerModel(
        input_dim=data_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Accuracy: {correct/total:.2%}")


model_path = "model/transformer.bin"
if not os.path.exists(model_path):
    create_model(model_path)
files = os.listdir("data/converted")
while True:
    random_file = files[np.random.randint(len(files))]
    print("Training on", random_file)
    train_model(model_path)
    print()
