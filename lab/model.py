import torch
import torch.nn as nn


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
data_dim = 6
sequence_length = 100
d_model = 256
nhead = 8
num_layers = 4
dim_feedforward = 4096

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def create_model(model_path: str):
    model = TransformerModel(
        input_dim=data_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    torch.save(model.state_dict(), model_path)


def load_model(model_path: str):
    # Initialize model
    model = TransformerModel(
        input_dim=data_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    return model
