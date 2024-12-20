import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class RowPercentageDifferenceScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # 对于这种基于行差值的变换，fit 不需要执行任何操作
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)  # 确保输入是 numpy 数组
        if X.shape[0] < 2:
            raise ValueError("The input data must have at least two rows.")

        # 计算相邻行的百分比差距
        percentage_diff = (X[1:] - X[:-1]) / np.abs(X[:-1])

        return percentage_diff


def preprocess(df: DataFrame) -> np.array:
    if "Adj Close" in df.columns:
        del df["Adj Close"]

    # Normalize date
    # df["Date Time"] = pd.to_datetime(df["Date Time"])
    # df = df.sort_values("Date Time")
    # date_begin = pd.to_datetime("2010-01-04 00:00:00")
    # date_end = pd.to_datetime("2022-12-30 00:00:00")
    # df["Date Time"] = (df["Date Time"] - date_begin) / (date_end - date_begin)

    # Delete date
    if "Date Time" in df.columns:
        del df["Date Time"]

    scaler = RowPercentageDifferenceScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data.astype(np.float32)
