from pandas import DataFrame
from numpy import ndarray
import numpy as np
import pandas as pd

# 定义归一化的起止日期
start_date = pd.to_datetime("2010-01-04")
end_date = pd.to_datetime("2022-12-30")
total_days = (end_date - start_date).days


def vectorize(df: DataFrame) -> ndarray:
    values = df.values

    # 将 Date Time 列转换为实数
    time = values[:, 0]
    time = [(t - start_date).days / total_days for t in time]
    time = np.array([time])

    # 数据归一化
    other = values[:, 1:]
    other = other / other.max(axis=0)

    ret = np.concatenate([time.T, other], axis=1).astype(np.float32)
    return ret
