import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

class HandoverEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # actuator 이름 기반으로 control index 가져오기
        self.arm_actuators = [self.model.actuator(f"act{i+1}").id for i in range(7)]
        self.gripper_actuator = self.model.actuator("gripper").id


        self.n_action = 8  # arm(7) + gripper(1)

        # actuator 실제 control range (중요)
        self.arm_ctrl_range = np.array([
            [-6.28, 6.28],
            [-2.09, 2.09],
            [-6.28, 6.28],
            [-0.19, 3.93],
            [-6.28, 6.28],
            [-1.69, 3.14],
            [-6.28, 6.28],
        ])
        self.gripper_ctrl_range = np.array([0.0, 255.0])

        # ✅ action_space를 실제 joint range로 정의
        self.action_space = gym.spaces.Box(
            low=np.concatenate((self.arm_ctrl_range[:, 0], [self.gripper_ctrl_range[0]])).astype(np.float32),
            high=np.concatenate((self.arm_ctrl_range[:, 1], [self.gripper_ctrl_range[1]])).astype(np.float32),
            dtype=np.float32
        )


        # observation: joint pos (7) + object pos (3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        init_qpos = np.array([0, -0.143, 0, 0.314, 0, 0, 0])
        self.data.qpos[:7] = init_qpos

        default_obj_pos = self.data.qpos[7:10].copy()
        delta = np.random.uniform(low=-0.005, high=0.005, size=3)
        self.data.qpos[7:10] = default_obj_pos + delta

        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        self.data.ctrl[:] = np.zeros_like(self.data.ctrl)  # ✅ ctrl은 모두 0으로 초기화

        mujoco.mj_forward(self.model, self.data)

        print("Current qpos after reset:", self.data.qpos[:7])
        return self._get_obs(), {}



    def step(self, action):
        assert len(action) == self.n_action

        arm_action = action[:7]
        gripper_action = action[7]

        self.data.ctrl[self.arm_actuators] = arm_action
        self.data.ctrl[self.gripper_actuator] = gripper_action

        mujoco.mj_step(self.model, self.data)

        reward = self._compute_reward()
        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        arm_pos = self.data.qpos[:7]
        obj_pos = self.data.xpos[self.model.body('object').id]

        ee_site = self.model.site('link_tcp').id  
        ee_pos = self.data.site_xpos[ee_site]
        rel_pos = obj_pos - ee_pos

        obs = np.concatenate([arm_pos, obj_pos, rel_pos])

        return obs.astype(np.float32)

    def _compute_reward(self):
        ee_site = self.model.site('link_tcp').id  
        ee_pos = self.data.site_xpos[ee_site]
        obj_pos = self.data.xpos[self.model.body('object').id]

        dist = np.linalg.norm(ee_pos - obj_pos)
        reward = -5.0 * dist  # distance shaping은 초기에 꽤 강하게 유지

        if dist < 0.05:
            reward += 3.0  # 접근 성공시 보상 더 강화

        gripper_closed = self.data.qpos[7] > 100

        if dist < 0.05 and gripper_closed:
            reward += 5.0

        lift_height = obj_pos[2]
        reward += 4.0 * max(lift_height - 0.05, 0)

        if lift_height > 0.1:
            reward += 10.0

        return reward

    def render(self):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        while viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            viewer.sync()


if __name__ == "__main__":
    env = HandoverEnv("./ufactory_xarm7/scene.xml")
    obs, _ = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # 랜덤 액션
        obs, reward, terminated, truncated, info = env.step(action)

    env.render()
