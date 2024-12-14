import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import create_model, load_model, device, sequence_length
from dataset import StockDataset


# Hyperparameters
batch_size = 512
epochs = 10
lr = 0.0001


def load_dataset(data_names: list[str], percent=1.0):
    datasets = []
    for data_name in tqdm(data_names, leave=False, desc="Loading datasets"):
        df = pd.read_csv(f"data/converted/{data_name}")
        dataset = StockDataset(df, sequence_length, device=device, percent=percent)
        datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets)


def train_model(model, dataset):
    # Create dataloaders
    test_size = len(dataset) // 10
    train_size = len(dataset) - test_size
    print(f"Train size: {train_size}, Test size: {test_size}")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print average and std of dataset
    avg = torch.mean(torch.tensor([y for _, y in dataset]))
    std = torch.std(torch.tensor([y for _, y in dataset]))
    print(f"Average: {avg:.20f}, Std: {std:.20f}")

    # Training loop
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in tqdm(
            train_loader, leave=False, desc=f"Epoch {epoch+1:02}/{epochs}"
        ):
            optimizer.zero_grad()
            outputs = model(x_batch)
            if x_batch.size(0) == 1:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        x_0, y_total = next(iter(test_loader))
        y_predict = model(x_0)
        if x_0.size(0) == 1:
            y_predict = y_predict.unsqueeze(0)
        for x_batch, y_batch in test_loader:
            if x_batch.size(0) == 0 or y_batch.size(0) == 0:
                continue
            outputs = model(x_batch)
            if x_batch.size(0) == 1:
                outputs = outputs.unsqueeze(0)
            y_total = torch.cat((y_total, y_batch))
            y_predict = torch.cat((y_predict, outputs))
        avg = torch.mean(y_predict)
        std = torch.std(y_predict)
        min = torch.min(y_predict)
        max = torch.max(y_predict)
        loss = criterion(y_predict, y_total)
        print(
            f"Epoch {epoch+1:02}/{epochs}, Loss: {loss:.5f}, Average: {avg:.20f}, Std: {std:.20f}, Min: {min:.20f}, Max: {max:.20f}"
        )

        # Write log
        log_path = "log/mlp-train.log"
        datetime = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{datetime}, {loss:.5f}, {avg:.20f}, {std:.20f}, {min:.20f}, {max:.20f}\n")

        # Failed if std is low
        if std < 0.08:
            print("Failed due to low std")
            return

    # Save
    torch.save(model.state_dict(), model_path)


model_path = "model/mlp.bin"
if not os.path.exists(model_path):
    create_model(model_path)
model = load_model(model_path)
files = os.listdir("data/converted")
dataset = load_dataset(files, percent=0.1)
train_model(model, dataset)
print()
