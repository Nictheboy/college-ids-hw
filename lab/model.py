import torch
import torch.nn as nn

# Hyperparameters
data_dim = 5
sequence_length = 33
d_model = 2048
n_layers = 6


class MlpModel(nn.Module):
    def __init__(self, input_length, input_dim, d_model, n_layers):
        super(MlpModel, self).__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_length * input_dim, d_model)
        self.mlp = nn.Sequential(
            *[nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU()) for _ in range(n_layers)]
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x = self.input_projection(x)
        x = self.input_projection(x.view(-1, self.input_length * self.input_dim))
        x = self.mlp(x)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def create_model(model_path: str):
    model = MlpModel(
        input_length=sequence_length,
        input_dim=data_dim,
        d_model=d_model,
        n_layers=n_layers,
    )
    torch.save(model.state_dict(), model_path)


def load_model(model_path: str):
    # Initialize model
    model = MlpModel(
        input_length=sequence_length,
        input_dim=data_dim,
        d_model=d_model,
        n_layers=n_layers,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    return model
