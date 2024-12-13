import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import create_model, load_model, device
from dataset import StockDataset


# Hyperparameters
sequence_length = 34
batch_size = 128
epochs = 5
lr = 0.001


def load_dataset(data_names: list[str], percent=1.0):
    datasets = []
    for data_name in tqdm(data_names, leave=False, desc="Loading datasets"):
        df = pd.read_csv(f"data/converted/{data_name}")
        dataset = StockDataset(df, sequence_length, device=device, percent=percent)
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)


def train_model(model, dataset):
    # Create dataloaders
    test_size = int(min(10000, 0.20 * len(dataset)))
    train_size = len(dataset) - test_size
    print(f"Train size: {train_size}, Test size: {test_size}")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation
    model.eval()
    correct_before = 0
    total_before = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            predictions = (outputs > 0.5).float()
            correct_before += (torch.abs(predictions - y_batch) < y_batch * 0.1).sum().item()
            total_before += y_batch.size(0)

    print(f"Accuracy Before: {correct_before/total_before:.2%}")

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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Save
    torch.save(model.state_dict(), model_path)

    # Evaluation
    model.eval()
    correct_after = 0
    total_after = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            predictions = (outputs > 0.5).float()
            correct_after += (torch.abs(predictions - y_batch) < y_batch * 0.1).sum().item()
            total_after += y_batch.size(0)

    print(f"Accuracy After: {correct_after/total_after:.2%}")

    # Write log
    log_path = "log/transformer-train.log"
    datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(
            f"{datetime}, {correct_before/total_before:.2%}, {correct_after/total_after:.2%}, {total_loss/len(train_loader):.4f}\n"
        )


model_path = "model/transformer.bin"
if not os.path.exists(model_path):
    create_model(model_path)
model = load_model(model_path)
files = os.listdir("data/converted")
dataset = load_dataset(files, percent=0.01)
train_model(model, dataset)
print()
