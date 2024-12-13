from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from model import load_model, device


class Predictor:
    def __init__(self):
        # 加载训练好的模型
        self.model = load_model("model/transformer.bin")
        self.model.eval()

        self.hold = False
        self.y_sum = 0
        self.y_count = 0

    def predict(self, df: DataFrame) -> int:
        # Normalize date
        df["Date Time"] = pd.to_datetime(df["Date Time"])
        df = df.sort_values("Date Time")
        date_begin = pd.to_datetime("2010-01-04 00:00:00")
        date_end = pd.to_datetime("2022-12-30 00:00:00")
        df["Date Time"] = (df["Date Time"] - date_begin) / (date_end - date_begin)

        scaler = MinMaxScaler()
        x = np.array([scaler.fit_transform(df)])
        y = self.model(torch.tensor(x, dtype=torch.float32, device=device)).item()
        self.y_sum += y
        self.y_count += 1
        hold_next = y > self.y_sum / self.y_count
        if self.hold == hold_next:
            ret = 0
        elif self.hold:
            ret = -1
        else:
            ret = 1
        self.hold = hold_next
        return ret
