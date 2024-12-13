import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random

from model import create_model, load_model, device
from dataset import StockDataset


# Hyperparameters
sequence_length = 34
batch_size = 128
epochs = 1
lr = 0.005


def load_dataset(data_names: list[str], percent=1.0):
    datasets = []
    for data_name in tqdm(data_names, leave=False, desc="Loading datasets"):
        df = pd.read_csv(f"data/converted/{data_name}")
        dataset = StockDataset(df, sequence_length, device=device, percent=percent)
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)


def train_model(model, dataset):
    # Create dataloaders
    train_size = len(dataset)
    print(f"Train size: {train_size}")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}"):
            if x_batch.size(0) == 0:
                continue
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.5f}")
        # Write log
        log_path = "log/transformer-train.log"
        datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{datetime}, {total_loss/len(train_loader):.5f}\n")

    # Save
    torch.save(model.state_dict(), model_path)


model_path = "model/transformer.bin"
if not os.path.exists(model_path):
    create_model(model_path)
model = load_model(model_path)
files = os.listdir("data/converted")
files = random.sample(files, 10)
dataset = load_dataset(files, percent=0.1)
train_model(model, dataset)
print()
