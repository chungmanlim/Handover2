import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

class HandoverEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # control index
        self.arm_actuators = [self.model.actuator(f"act{i+1}").id for i in range(7)]
        self.gripper_actuator = self.model.actuator("gripper").id

        self.arm_ctrl_range = np.array([
            [-6.28, 6.28],  # act1 
            [-2.09, 2.09],  # act2
            [-6.28, 6.28],  # act3
            [-0.19, 3.93],  # act4
            [-6.28, 6.28],  # act5
            [-1.69, 3.14],  # act6
            [-6.28, 6.28]   # act7
        ])

        self.gripper_ctrl_range = np.array([0.0, 255.0]) # gripper

        # action_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # observation
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Arm 초기 포즈 (약간 랜덤)
        init_arm_qpos = np.array([0, 0, 0, 0, 0, 0, 0])
        arm_noise = np.random.uniform(low=-0.1, high=0.1, size=7)
        self.data.qpos[:7] = init_arm_qpos + arm_noise

        # Gripper 초기화 (left/right driver fully open)
        self.data.qpos[7]  = 0.0   # left_driver_joint
        self.data.qpos[10] = 0.0   # right_driver_joint

        # Object 초기 위치 (pos = qpos[13:16])
        object_init_pos = np.array([0.6, 0.0, 0.0])
        delta_xy = np.random.uniform(low=-0.05, high=0.05, size=2)  # 5cm noise
        delta_z = np.random.uniform(low=0.00, high=0.02, size=1)    # 살짝 위로
        delta = np.concatenate([delta_xy, delta_z])
        self.data.qpos[13:16] = object_init_pos + delta

        # Object orientation (quat)
        self.data.qpos[16:20] = np.array([1.0, 0.0, 0.0, 0.0])  # no rotation

        # Velocity 초기화
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0

        mujoco.mj_forward(self.model, self.data)

        # 디버그 출력
        # print("object qpos (pos):", self.data.qpos[13:16])
        # print("object xpos (current pos):", self.data.xpos[self.model.body('object').id])
        # print("gripper qpos:", self.data.qpos[7], self.data.qpos[10])

        return self._get_obs(), {}
    

    def step(self, action):
        # arm part rescale
        arm_action = self._rescale_action(action[:7], self.arm_ctrl_range)

        # gripper part rescale
        gripper_action = (action[7] + 1.0) * 0.5 * 255.0

        self.data.ctrl[self.arm_actuators] = arm_action
        self.data.ctrl[self.gripper_actuator] = gripper_action

        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        reward, terminated = self._compute_reward()
        # terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}



    def _rescale_action(self, action, ctrl_range):
        low, high = ctrl_range[:, 0], ctrl_range[:, 1]
        return low + (action + 1.0) * 0.5 * (high - low)


    def _get_obs(self):
        # Normalize arm pos
        arm_pos = (self.data.qpos[:7] - self.arm_ctrl_range[:, 0]) / (self.arm_ctrl_range[:, 1] - self.arm_ctrl_range[:, 0])
        arm_pos = np.clip(arm_pos, 0.0, 1.0)

        # object pos
        obj_pos = self.data.xpos[self.model.body('object').id]

        # Position of grippers
        left_tip = self.data.site_xpos[self.model.site('left_finger_tip_site').id]
        right_tip = self.data.site_xpos[self.model.site('right_finger_tip_site').id]
        ee_pos = 0.5 * (left_tip + right_tip)

        rel_obj_pos = obj_pos - ee_pos

        # gripper → joint7, joint10 → range [0, 0.85]
        gripper_left_qpos = self.data.qpos[7]    # left_driver_joint
        gripper_right_qpos = self.data.qpos[10]  # right_driver_joint
        gripper_pos = (gripper_left_qpos + gripper_right_qpos) * 0.5

        # normalize → range [0, 0.85]
        gripper_max = 0.85
        gripper_pos_norm = np.clip(gripper_pos / gripper_max, 0.0, 1.0)

        # concat
        obs = np.concatenate([arm_pos, obj_pos, ee_pos, rel_obj_pos, [gripper_pos_norm]])
        return obs.astype(np.float32)


    def _compute_reward(self):
        # Position of grippers
        left_tip = self.data.site_xpos[self.model.site('left_finger_tip_site').id]
        right_tip = self.data.site_xpos[self.model.site('right_finger_tip_site').id]
        ee_pos = 0.5 * (left_tip + right_tip)

        obj_pos = self.data.xpos[self.model.body('object').id]

        dist = np.linalg.norm(ee_pos - obj_pos)

        reward = np.exp(-dist) - 1.0

        if dist < 0.05:
            reward += 10
            terminated = True
        else:
            terminated = False

        # print(f"ee_pos: {ee_pos}, obj_pos: {obj_pos}, dist: {dist}")
        # print(f"dist={dist:.3f}, effort_penalty={effort_penalty:.3f}, reward={reward:.3f}")

        return reward, terminated


    def render(self):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        while viewer.is_running():
            mujoco.mj_step(self.model, self.data)
            viewer.sync()


if __name__ == "__main__":
    env = HandoverEnv("./ufactory_xarm7/scene.xml")
    obs, _ = env.reset()

    for _ in range(5):
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
