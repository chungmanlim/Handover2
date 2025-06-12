import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

from sb3_contrib import RecurrentPPO
from handover_env import HandoverEnv

env = HandoverEnv("./ufactory_xarm7/scene.xml")
model = RecurrentPPO.load("ppo_handover_grasp", env=env, device="cuda")

viewer = mujoco.viewer.launch_passive(env.model, env.data)
obs, _ = env.reset()

# ✅ lstm state 초기화 (버전 안정)
n_layers, n_envs, hidden_size = model.policy.lstm_hidden_state_shape

lstm_state = (
    np.zeros((n_layers, n_envs, hidden_size), dtype=np.float32),
    np.zeros((n_layers, n_envs, hidden_size), dtype=np.float32)
)
episode_start = np.ones((n_envs,), dtype=bool)

for _ in range(10000):
    action, lstm_state = model.predict(
        obs,
        state=lstm_state,
        episode_start=episode_start,
        deterministic=False
    )

    print("Action:", action)

    obs, reward, terminated, truncated, info = env.step(action)
    viewer.sync()

    if terminated or truncated:
        obs, _ = env.reset()
        lstm_state = (
            np.zeros((n_layers, n_envs, hidden_size), dtype=np.float32),
            np.zeros((n_layers, n_envs, hidden_size), dtype=np.float32)
        )
        episode_start = np.ones((n_envs,), dtype=bool)
    else:
        episode_start = np.zeros((n_envs,), dtype=bool)

env.render()

