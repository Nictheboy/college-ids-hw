from pandas import DataFrame
import numpy as np
import torch
from model import load_model, device, sequence_length
from preprocess import preprocess


class Predictor:
    def __init__(self):
        # 加载训练好的模型
        self.model = load_model("model/mlp.bin")
        self.model.eval()

    def predict(self, df: DataFrame) -> int:
        x = np.array([preprocess(df)[-sequence_length:]])
        y = self.model(torch.tensor(x, dtype=torch.float32, device=device)).item()
        return 1 if y > 0.5 else -1
