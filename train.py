import gymnasium as gym
import numpy as np
import mujoco
import time
import torch

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env

# ✅ 현재 만든 환경 import
from handover_env import HandoverEnv  # (당신이 만든 HandoverEnv 모듈 import)

# ✅ Mujoco 환경 로드
env = HandoverEnv("./ufactory_xarm7/scene.xml")

# ✅ (환경 체크 — 디버깅 단계에서만)
check_env(env)

# LSTM 정책 정의
policy_kwargs = dict(
    net_arch=[256, 128, 64], 
    lstm_hidden_size=512,
    n_lstm_layers=1
)

# policy_kwargs = dict(
#     net_arch=[],  # MLP 생략
#     lstm_hidden_size=256,
#     n_lstm_layers=3
# )

# ✅ PPO 모델 정의
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    device="cuda",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=20,
    ent_coef=0.05,
    policy_kwargs=policy_kwargs
)

# ✅ 학습 시작
model.learn(total_timesteps=2_000_000)

# ✅ 모델 저장
model.save("ppo_handover_grasp")
