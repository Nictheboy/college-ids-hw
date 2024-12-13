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
        self.y_list_7 = np.array([])
        self.y_list_30 = np.array([])

    def predict(self, df: DataFrame) -> int:
        # Normalize date
        df["Date Time"] = pd.to_datetime(df["Date Time"])
        df = df.sort_values("Date Time")
        date_begin = pd.to_datetime("2010-01-04 00:00:00")
        date_end = pd.to_datetime("2022-12-30 00:00:00")
        df["Date Time"] = (df["Date Time"] - date_begin) / (date_end - date_begin)

        scaler = MinMaxScaler()
        x = np.array([scaler.fit_transform(df[-100:])])
        y = self.model(torch.tensor(x, dtype=torch.float32, device=device)).item()
        self.y_list_7 = np.append(self.y_list_7, y)
        self.y_list_30 = np.append(self.y_list_30, y)
        if len(self.y_list_7) > 7:
            self.y_list_7 = self.y_list_7[1:]
        if len(self.y_list_30) > 30:
            self.y_list_30 = self.y_list_30[1:]
        hold_next = y > np.mean(self.y_list_30)
        # hold_next = y > 0.5 or np.mean(self.y_list_7) > 0.5 or np.mean(self.y_list_30) > 0.5
        if hold_next:
            ret = 1
        else:
            ret = -1
        return ret
