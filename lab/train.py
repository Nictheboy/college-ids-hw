import numpy as np
import pandas as pd
import torch.nn as nn
from gymnasium import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from vectorize import vectorize


# =============================
# 环境定义
# =============================
class YourTradingEnv(Env):
    def __init__(self, df: pd.DataFrame, window_size=34):
        """
        初始化交易环境
        :param df: pd.DataFrame, 包含市场数据 (列为 Open, High, Low, Close, Volume)
        :param window_size: 用于构建状态的时间窗口大小
        """
        super(YourTradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.current_step = window_size

        # 定义状态空间 (最近 window_size 天的 OHLCV 数据)
        self.observation_space = spaces.Box(low=0, high=1, shape=(window_size, 6), dtype=np.float32)

        # 定义动作空间 (-1: 卖出, 0: 持有, 1: 买入)
        self.action_space = spaces.Discrete(3)

        # 初始化账户状态
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0

    def reset(self, **kwargs):
        """重置环境"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0
        return self._get_observation(), {}

    def _get_observation(self):
        """获取当前状态 (过去 window_size 天的数据)"""
        return vectorize(self.df.iloc[self.current_step - self.window_size : self.current_step])

    def step(self, action):
        """执行一个动作"""
        done = False
        reward = 0

        # 获取当前开盘价
        current_open = self.df.iloc[self.current_step]["Open"]

        # 执行买入动作
        if action == 2:  # 买入
            if self.balance >= current_open:
                self.shares_held += 1
                self.balance -= current_open
        # 执行卖出动作
        elif action == 0:  # 卖出
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_open

        # 计算奖励 (账户总价值的变化)
        next_close = self.df.iloc[self.current_step]["Close"]
        portfolio_value = self.balance + self.shares_held * next_close
        reward = portfolio_value - (self.balance + self.shares_held * current_open)
        self.total_reward += reward

        # 移动到下一个时间步
        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        # 返回新的状态、奖励和完成标志
        return self._get_observation(), reward, done, done, {}


# =============================
# 数据准备和训练
# =============================
# 创建示例数据
df = pd.read_csv("data/converted/000001.SZ.csv")
df["Date Time"] = pd.to_datetime(df["Date Time"])
del df["Adj Close"]

# 初始化交易环境
env = DummyVecEnv([lambda: YourTradingEnv(df)])

# 创建 PPO 模型
# model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# 读取模型
model = PPO.load("model/ppo_trading", env, verbose=1, device="cpu")

# 训练模型
model.learn(total_timesteps=df.shape[0])

# 保存模型
model.save("model/ppo_trading")
