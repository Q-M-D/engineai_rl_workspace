import os, asyncio
import pygame
from threading import Thread
import numpy as np
from tqdm import tqdm

from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR

from engineai_rl_workspace.utils import (
    get_args,
    generate_cfg_files_from_json,
)
from engineai_rl_workspace.utils.process_resume_files import (
    get_log_root_and_log_dir,
)
from engineai_rl_workspace import (
    INITIALIZATION_COMPLETE_MESSAGE,
)
from collections import deque
import mujoco

def gymQuat2MujocoQuat(gym_quat):
    return np.array([gym_quat[3], gym_quat[0], gym_quat[1], gym_quat[2]])

def init_args(args):
    args.resume = True
    _, log_dir = get_log_root_and_log_dir(args)
    generate_cfg_files_from_json(args)
    from engineai_rl_workspace.utils.exp_registry import exp_registry

    policy_dir = os.path.join(log_dir, "policies")
    contact_estimator_policy_path = os.path.join(policy_dir, f"{args.exp_name}_{args.load_run}_AttentionNetwork.pt")
    actor_policy_path = os.path.join(policy_dir, f"{args.exp_name}_{args.load_run}_Mlp.pt")

    (
        args,
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        runner_class,
        algo_class,
        log_dir,
        log_root,
        env_cfg,
        algo_cfg,
    ) = exp_registry.get_class_and_cfg(name=args.exp_name, args=args)

    print(INITIALIZATION_COMPLETE_MESSAGE)
    return env_cfg, algo_cfg, contact_estimator_policy_path, actor_policy_path
    
    action_joints = env_cfg.env.action_joints
    init_state = env_cfg.init_state
    obs_scales = env_cfg.normalization.obs_scales
    clip_actions = env_cfg.normalization.clip_actions
    obs_list = env_cfg.obs_list
    control_cfg = env_cfg.control

class RLController:
    class ObsConfig:
        dof_pos = 12
        dof_vel = 12
        actions = 12
        base_ang_vel = 3
        base_euler_xyz = 3
        contact_mask = 2
        gait_phase = 2
        commands = 3
    def __init__(
        self,
        env_cfg,
        algo_cfg,
        contact_estimator_policy_path,
        actor_policy_path
    ):
        self.env_cfg = env_cfg
        self.algo_cfg = algo_cfg
        self.contact_estimator_policy_path = contact_estimator_policy_path
        self.actor_policy_path = actor_policy_path
        self.actor, self.contact_estimator = self.init_networks()
        self.init_obs()

    def init_networks(self):
        """
        contact_estimator:
            - input dim: 840
            - output dim: 2
        actor:
            - input dim: 980
            - output dim: 12
        """
        import torch
        actor = torch.jit.load(self.actor_policy_path)
        contact_estimator = torch.jit.load(self.contact_estimator_policy_path)
        return actor, contact_estimator

    def init_obs(self):
        
        self.contact_obs_deque = deque(maxlen=self.algo_cfg.input.components.contact.obs_history_length)
        self.actor_obs_deque = deque(maxlen=self.algo_cfg.input.components.actor.obs_history_length)
        
        self.contact_obs_len = sum(
            getattr(self.ObsConfig, item, 0)
            for item in self.algo_cfg.input.components.contact.obs_list
        )
        self.actor_obs_len = sum(
            getattr(self.ObsConfig, item, 0)
            for item in self.algo_cfg.input.components.actor.obs_list
        ) + self.ObsConfig.gait_phase + self.ObsConfig.commands

        # Pre-fill observation deques with zeros efficiently
        contact_zero = np.zeros(self.contact_obs_len)
        actor_zero = np.zeros(self.actor_obs_len)
        self.contact_obs_deque.extend([contact_zero.copy() for _ in range(self.algo_cfg.input.components.contact.obs_history_length)])
        self.actor_obs_deque.extend([actor_zero.copy() for _ in range(self.algo_cfg.input.components.actor.obs_history_length)])

    def get_actions(self, t, data):
        pass
    
    def get_init_state(self, model):
        root_pos = self.env_cfg.init_state.pos
        root_quat = gymQuat2MujocoQuat(self.env_cfg.init_state.rot)
        joint_pos = []
        for i in range(model.njnt-1):
            joint_pos.append(self.env_cfg.init_state.default_joint_angles[model.joint(i+1).name])
        
        root_vel = self.env_cfg.init_state.lin_vel
        root_ang_vel = self.env_cfg.init_state.ang_vel
        joint_vel = [0.0 for _ in range(model.nv-6)]
        
        return np.concatenate([root_pos, root_quat, joint_pos]), np.concatenate([root_vel, root_ang_vel, joint_vel])
    
    


def set_commands_from_joystick(env, x_vel_cmd, y_vel_cmd, yaw_vel_cmd):
    global last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
    env.commands[:, 0] = x_vel_cmd
    env.commands[:, 1] = y_vel_cmd
    env.commands[:, 2] = yaw_vel_cmd
    if (
        last_x_vel_cmd != x_vel_cmd
        or last_y_vel_cmd != y_vel_cmd
        or last_yaw_vel_cmd != yaw_vel_cmd
    ):
        print("Current command: ", env.commands[:, :3])
        last_x_vel_cmd = x_vel_cmd
        last_y_vel_cmd = y_vel_cmd
        last_yaw_vel_cmd = yaw_vel_cmd
    return env.goal_dict


def use_joystick(args):
    joystick_opened = False
    pygame.init()
    try:
        # get joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"Unable to turn on joystick：{e}")

    # handle joystick thread
    def handle_joystick_input():
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
        while True:
            # get joystick input
            pygame.event.get()

            # update command
            x_vel_cmd = -joystick.get_axis(1) * args.joystick_scale[0]
            y_vel_cmd = -joystick.get_axis(0) * args.joystick_scale[1]
            yaw_vel_cmd = -joystick.get_axis(3) * args.joystick_scale[2]

            # wait for a short period of time
            pygame.time.delay(100)

    # start thread
    if joystick_opened:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


async def main(controller: RLController):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
    (
        x_vel_cmd,
        y_vel_cmd,
        yaw_vel_cmd,
        last_x_vel_cmd,
        last_y_vel_cmd,
        last_yaw_vel_cmd,
    ) = (0, 0, 0, 0, 0, 0)
    print("Start simulating")
    # Get model path from controller config and resolve the real path
    # model_path = controller.env_cfg.asset.xmlfile.format(ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR)
    model_path = '/home/mmlab-rl/codes/engineai_rl_workspace/engineai_gym/engineai_gym/resources/robots/biped/dora2/mjcf/dora2_shoes.xml'
    # Initialize MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Set initial state
    data.qpos[:], data.qvel[:] = controller.get_init_state(model)
    mujoco.mj_forward(model, data)

    # Create visualization if enabled
    if VIS:
        cam = mujoco.MjvCamera()
        cam.distance = 3.0
        cam.azimuth = 90
        cam.elevation = -20
        scene = mujoco.MjvScene(model, maxgeom=100)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, cam, scene)
    
    
    
    
    

if __name__ == "__main__":
    VIS = False
    args = get_args()
    if args.use_joystick:
        use_joystick(args)
    env_cfg, algo_cfg, contact_estimator_policy_path, actor_policy_path = init_args(args)
    controller = RLController(env_cfg, algo_cfg, contact_estimator_policy_path, actor_policy_path)
    asyncio.run(main(controller))