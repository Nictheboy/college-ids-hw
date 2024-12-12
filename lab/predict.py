from pandas import DataFrame
from stable_baselines3 import PPO
from vectorize import vectorize

# 加载训练好的模型
model = PPO.load("model/ppo_trading", device="cpu")


def predict(data: DataFrame) -> int:
    action, _ = model.predict(vectorize(data))
    return action - 1
