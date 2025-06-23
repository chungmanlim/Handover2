import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from handover_env import HandoverEnv

# ✅ 환경 로드
env = HandoverEnv("./ufactory_xarm7/scene.xml")

# ✅ MLP PPO 모델 로드
model = PPO.load("ppo_handover_grasp", env=env, device="cuda")

# ✅ Viewer 띄우기
viewer = mujoco.viewer.launch_passive(env.model, env.data)
obs, _ = env.reset()

# ✅ reward 기록용 리스트
reward_list = []
episode_reward = 0.0

# ✅ 테스트 루프
for step in range(10000):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)

    viewer.sync()

    # ✅ reward 기록
    episode_reward += reward
    reward_list.append(episode_reward)

    # ✅ episode 끝났을 때
    if terminated or truncated:
        print(f"Episode done at step {step}, total reward: {episode_reward:.2f}")
        obs, _ = env.reset()
        episode_reward = 0.0

env.render()

# ✅ reward 그래프 그리기
plt.figure(figsize=(10, 4))
plt.plot(reward_list)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Test Episode Reward Trend")
plt.grid()
plt.show()