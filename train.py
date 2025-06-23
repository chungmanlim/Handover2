import gymnasium as gym
import numpy as np
import mujoco
import time
import torch

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env

# ✅ 현재 만든 환경 import
from handover_env import HandoverEnv

# ✅ Mujoco 환경 로드
env = HandoverEnv("./ufactory_xarm7/scene.xml")

check_env(env)

policy_kwargs = dict(
    net_arch=[256, 256, 256, 256],  # 4 layers, each 256 units
    log_std_init=0.0
)

# ✅ PPO 모델 정의
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=20,
    ent_coef=0.1,
    policy_kwargs=policy_kwargs
)

# ✅ 학습 시작
model.learn(total_timesteps=1_000_000)

# ✅ 모델 저장
model.save("ppo_handover_grasp")
