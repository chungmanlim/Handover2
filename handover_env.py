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


        # ✅ action space를 줄임
        self.n_action = 4  # act2, act4, act6, gripper

        self.arm_ctrl_range = np.array([
            [-2.09, 2.09],   # act2
            [-0.19, 3.93],   # act4
            [-1.69, 3.14],   # act6
            [-6.28, 6.28]   # act7
        ])

        self.gripper_ctrl_range = np.array([0.0, 255.0])

        # action space: 4 continuous + 1 discrete
        self.action_space = gym.spaces.Box(
            low=np.array([
                self.arm_ctrl_range[0][0],  # act2
                self.arm_ctrl_range[1][0],  # act4
                self.arm_ctrl_range[2][0],  # act6
                self.arm_ctrl_range[3][0],  # act7
                0.0  # gripper (continuous 0~1로 바꿔줌)
            ]),
            high=np.array([
                self.arm_ctrl_range[0][1],
                self.arm_ctrl_range[1][1],
                self.arm_ctrl_range[2][1],
                self.arm_ctrl_range[3][1],
                1.0
            ]),
            dtype=np.float32
        )

        # observation: joint pos (7) + object pos (3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        init_qpos = np.array([0, 0.724, 0, 1.11, 0, 0, 0])
        self.data.qpos[:7] = init_qpos

        default_obj_pos = self.data.qpos[7:10].copy()
        delta = np.random.uniform(low=-0.005, high=0.005, size=3)
        self.data.qpos[7:10] = default_obj_pos + delta

        self.data.qvel[:] = np.zeros_like(self.data.qvel)
        self.data.ctrl[:] = np.zeros_like(self.data.ctrl)  # ✅ ctrl은 모두 0으로 초기화

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}



    def step(self, action):
        arm_action = action[:4]
        gripper_action = action[4]

        full_arm_action = np.array([
            0.0,               # act1 고정
            arm_action[0],     # act2
            0.0,               # act3 고정
            arm_action[1],     # act4
            0.0,               # act5 고정
            arm_action[2],     # act6
            arm_action[3],     # act7
        ])

        self.data.ctrl[self.arm_actuators] = full_arm_action
        self.data.ctrl[self.gripper_actuator] = gripper_action * 255.0  # 0~1 → 0~255

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
        reward = -10.0 * dist  # 접근 유도 보상

        # 1단계: 근접하면 보상
        if dist < 0.05:
            reward += 5.0

            # 2단계: 근접 시 그립퍼 닫기 유도 보상
            gripper_pos = self.data.qpos[7]
            reward += (gripper_pos / 255.0) * 5.0

        gripper_closed = self.data.qpos[7] > 200  
        if dist < 0.05 and gripper_closed:
            reward += 5.0

        # 3단계: 들어올리면 추가 보상
        lift_height = obj_pos[2]
        reward += 4.0 * max(lift_height - 0.05, 0)

        # ✨ 수직 lifting shaping 추가
        xy_offset = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        reward += 5.0 * (1 - xy_offset)  # 물체 중심으로 올릴수록 보상

        if lift_height > 0.1 and xy_offset < 0.02:
            reward += 10.0  # 안정적으로 위에서 잘 들었을 때 보상 강화

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
