import os
import asyncio
import pygame
from threading import Thread
import numpy as np
from tqdm import tqdm
from git import Repo

from engineai_gym import ENGINEAI_GYM_PACKAGE_DIR
from engineai_gym.tester.tester import Tester
from engineai_rl_workspace.utils import (
    get_args,
    generate_cfg_files_from_json,
)
from engineai_rl_workspace.utils.process_resume_files import (
    get_log_root_and_log_dir,
    checkout_resume_commit,
)
from engineai_rl_workspace import (
    REDIS_HOST,
    LOCK_KEY,
    REDIS_PORT,
    LOCK_TIMEOUT,
    LOCK_MESSAGE,
    INITIALIZATION_COMPLETE_MESSAGE,
    ENGINEAI_WORKSPACE_ROOT_DIR,
)
from engineai_rl_lib.redis_lock import RedisLock
from engineai_rl_lib.git import (
    get_current_commit_and_branch,
    checkout_commit_or_branch,
    unstash_files,
    apply_patch,
)
from collections import deque
import mujoco
import mujoco.viewer
import time
from scipy.spatial.transform import Rotation as R
import torch


def gymQuat2MujocoQuat(gym_quat):
    return np.array([gym_quat[3], gym_quat[0], gym_quat[1], gym_quat[2]])


def quaternion2Euler(quat):
    r = R.from_quat(quat)
    return r.as_euler('xyz')


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
        self.cycle_time = self.env_cfg.gait.cycle_time
        self.last_action = np.zeros(self.ObsConfig.actions)
        self.init_pd_controller()
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

    def get_init_state(self, model):
        root_pos = self.env_cfg.init_state.pos
        root_quat = gymQuat2MujocoQuat(self.env_cfg.init_state.rot)
        self.default_joint_angles = []
        for i in range(model.njnt-1):
            self.default_joint_angles.append(
                self.env_cfg.init_state.default_joint_angles[model.joint(i+1).name])

        root_vel = self.env_cfg.init_state.lin_vel
        root_ang_vel = self.env_cfg.init_state.ang_vel
        joint_vel = [0.0 for _ in range(model.nv-6)]

        return np.concatenate([root_pos, root_quat, self.default_joint_angles]), np.concatenate([root_vel, root_ang_vel, joint_vel])

    def init_obs(self):
        self.contact_obs_deque = deque(
            maxlen=self.algo_cfg.input.components.contact.obs_history_length)
        self.actor_obs_deque = deque(
            maxlen=self.algo_cfg.input.components.actor.obs_history_length)

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
        self.contact_obs_deque.extend([contact_zero.copy() for _ in range(
            self.algo_cfg.input.components.contact.obs_history_length)])
        self.actor_obs_deque.extend([actor_zero.copy() for _ in range(
            self.algo_cfg.input.components.actor.obs_history_length)])

    def init_pd_controller(self):
        keys = [
            'hip_roll_joint',
            'hip_yaw_joint',
            'hip_pitch_joint',
            'knee_joint',
            'ankle_pitch_joint',
            'ankle_roll_joint'
        ]
        # Initialize proportional gains (kp) for each joint in both legs
        self.kp = [self.env_cfg.control.stiffness[joint] for joint in keys] * 2
        # Initialize derivative gains (kd) for each joint in both legs
        self.kd = [self.env_cfg.control.damping[joint] for joint in keys] * 2

    def get_phase(self, t):
        phase = t / self.cycle_time
        return phase

    def get_actions(self, t, data, commands):
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        omega = data.sensor('angular-velocity').data.astype(np.double)

        obs_list = {
            'dof_pos':          data.qpos[7:].copy() * self.env_cfg.normalization.obs_scales['dof_pos'],
            'dof_vel':          data.qvel[6:].copy() * self.env_cfg.normalization.obs_scales['dof_vel'],
            'actions':          self.last_action.copy() * 1,
            'base_ang_vel':     omega * self.env_cfg.normalization.obs_scales['base_ang_vel'],
            'base_euler_xyz':   quaternion2Euler(quat) * self.env_cfg.normalization.obs_scales['base_euler_xyz'],
            'contact_mask':     np.array([0, 0]) * 1,
        }

        contact_obs = np.concatenate([
            obs_list[item] for item in self.algo_cfg.input.components.contact.obs_list
        ])

        self.contact_obs_deque.append(contact_obs)
        contact_input = np.concatenate(self.contact_obs_deque)
        estimated_contact, attention_weight = self.contact_estimator(
            torch.tensor(contact_input).float().unsqueeze(0))

        obs_list['contact_mask'] = estimated_contact.detach().cpu().numpy().flatten()

        phase = self.get_phase(t)
        goal = np.concatenate([
            [np.sin(2 * np.pi * phase),
             np.cos(2 * np.pi * phase)],
            commands
        ])

        actor_obs = np.concatenate([
            obs_list[item] for item in self.algo_cfg.input.components.actor.obs_list
        ] + [goal])

        self.actor_obs_deque.append(actor_obs)
        actor_input = np.concatenate(self.actor_obs_deque)

        # Pass actor_obs to the actor network and update last_action
        action = self.actor(torch.tensor(actor_input).float().unsqueeze(0))
        self.last_action = action.detach().cpu().numpy().flatten()
        return self.last_action

    def get_torque(self, actions, q, dq):
        """PD controller"""
        cliped_actions = np.clip(
            actions, -self.env_cfg.normalization.clip_actions, self.env_cfg.normalization.clip_actions)
        scaled_actions = cliped_actions * self.env_cfg.control.action_scales['joint']

        return (scaled_actions + self.default_joint_angles - q) * self.kp - dq * self.kd


def set_cam_viewer(viewer):
    viewer.cam.lookat = [0, 0, 0]  # 摄像机视线的目标点
    viewer.cam.distance = 4.0      # 摄像机与目标点的距离
    viewer.cam.elevation = -20     # 摄像机仰角
    viewer.cam.azimuth = 135       # 摄像机方位角


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
            # Initialize last command variables if not already set
            # Initialize last command variables if not already set
            if 'last_x_vel_cmd' not in globals():
                last_x_vel_cmd = x_vel_cmd
            if 'last_y_vel_cmd' not in globals():
                last_y_vel_cmd = y_vel_cmd
            if 'last_yaw_vel_cmd' not in globals():
                last_yaw_vel_cmd = yaw_vel_cmd
            if (
                x_vel_cmd != last_x_vel_cmd
                or y_vel_cmd != last_y_vel_cmd
                or yaw_vel_cmd != last_yaw_vel_cmd
            ):
                print(
                    f"Joystick command: x_vel_cmd={x_vel_cmd:.2f}, y_vel_cmd={y_vel_cmd:.2f}, yaw_vel_cmd={yaw_vel_cmd:.2f}"
                )
                last_x_vel_cmd = x_vel_cmd
                last_y_vel_cmd = y_vel_cmd
                last_yaw_vel_cmd = yaw_vel_cmd

            # wait for a short period of time
            pygame.time.delay(100)

    # start thread
    if joystick_opened:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()


async def main(args):
    global lock, repo, current_commit, current_branch
    lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, last_x_vel_cmd, last_y_vel_cmd, last_yaw_vel_cmd
    (
        x_vel_cmd,
        y_vel_cmd,
        yaw_vel_cmd,
        last_x_vel_cmd,
        last_y_vel_cmd,
        last_yaw_vel_cmd,
    ) = (0, 0, 0, 0, 0, 0)

    try:
        args.resume = True
        if not await lock.acquire():
            print("Could not acquire lock, exiting...")
            return
        repo = Repo(ENGINEAI_WORKSPACE_ROOT_DIR)
        current_commit, current_branch = get_current_commit_and_branch(repo)
        _, log_dir = get_log_root_and_log_dir(args)
        checkout_resume_commit(log_dir, repo)
        apply_patch(os.path.join(log_dir, "resume.patch"), ENGINEAI_WORKSPACE_ROOT_DIR)
        generate_cfg_files_from_json(args)
        from engineai_gym.wrapper import VecGymWrapper, RecordVideoWrapper
        import engineai_rl_workspace.exps
        from engineai_rl_workspace.utils.exp_registry import exp_registry

        policy_dir = os.path.join(log_dir, "policies")
        contact_estimator_policy_path = os.path.join(
            policy_dir, f"{args.exp_name}_{args.load_run}_AttentionNetwork.pt")
        actor_policy_path = os.path.join(
            policy_dir, f"{args.exp_name}_{args.load_run}_Mlp.pt")

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
        controller = RLController(
            env_cfg, algo_cfg, contact_estimator_policy_path, actor_policy_path)

        print("Start simulating")
        # Get model path from controller config and resolve the real path
        # model_path = controller.env_cfg.asset.xmlfile.format(ENGINEAI_GYM_PACKAGE_DIR=ENGINEAI_GYM_PACKAGE_DIR)
        model_path = '/home/mmlab-rl/codes/engineai_rl_workspace/engineai_gym/engineai_gym/resources/robots/biped/dora2/mjcf/dora2_shoes.xml'
        # Initialize MuJoCo model and data
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # set simulation parameters
        dt = model.opt.timestep = controller.env_cfg.sim.dt
        decimation = controller.env_cfg.control.decimation

        sim_time = 0.0

        # Set initial state
        data.qpos[:], data.qvel[:] = controller.get_init_state(model)
        mujoco.mj_forward(model, data)
        actions = np.zeros(controller.ObsConfig.actions)
        counter = 0

        # Create visualization if enabled

        if VIS:
            cam = mujoco.MjvCamera()
            cam.distance = 3.0
            cam.azimuth = 90
            cam.elevation = -20

            # 使用launch_passive启动查看器（在macOS上需要使用mjpython运行）
            with mujoco.viewer.launch_passive(model, data) as viewer:
                set_cam_viewer(viewer)
                start = time.time()

                while viewer.is_running():
                    step_start = time.time()

                    if counter % decimation == 0:
                        actions = controller.get_actions(
                            sim_time, data, [x_vel_cmd, y_vel_cmd, yaw_vel_cmd])
                    counter += 1
                    data.ctrl = controller.get_torque(
                        actions, data.qpos[7:], data.qvel[6:])

                    sim_time += dt

                    # 执行物理模拟步骤
                    mujoco.mj_step(model, data)

                    # 同步查看器
                    viewer.sync()

                    # 基本时间控制，相对于挂钟时间会有漂移
                    time_until_next_step = model.opt.timestep * \
                        2 - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while True:
                step_start = time.time()
                if counter % decimation == 0:
                    actions = controller.get_actions(
                        sim_time, data, [x_vel_cmd, y_vel_cmd, yaw_vel_cmd])
                counter += 1
                data.ctrl = controller.get_torque(actions, data.qpos[7:], data.qvel[6:])

                sim_time += dt

                mujoco.mj_step(model, data)

                print(time.time())
                # 基本时间控制，相对于挂钟时间会有漂移
                time_until_next_step = model.opt.timestep * \
                    10 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except Exception as e:
        print(f"Exception in main: {e}")
        raise
    finally:
        # Cleanup logic that should always run
        print("Cleaning up simulation...")
        if 'lock' in globals() and hasattr(lock, 'redis') and lock.redis.get(lock.lock_key) == lock.pid.encode():
            try:
                if 'repo' in globals() and 'current_commit' in globals() and 'current_branch' in globals():
                    checkout_commit_or_branch(repo, current_commit, current_branch)
                    unstash_files(repo)
            except Exception as cleanup_error:
                print(f"Error during git cleanup: {cleanup_error}")
            finally:
                lock.release()


if __name__ == "__main__":
    global lock, repo, current_commit, current_branch
    try:
        VIS = True
        args = get_args()
        if args.use_joystick:
            use_joystick(args)
        asyncio.run(main(args))
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"Script interrupted: {type(e).__name__}: {e}")
        if lock and hasattr(lock, 'redis') and lock.redis.get(lock.lock_key) == lock.pid.encode():
            try:
                if repo and current_commit is not None and current_branch is not None:
                    checkout_commit_or_branch(repo, current_commit, current_branch)
                    unstash_files(repo)
                    print("Git state restored successfully")
            except Exception as cleanup_error:
                print(f"Error during git cleanup: {cleanup_error}")
            finally:
                try:
                    lock.release()
                    print("Redis lock released")
                except Exception as lock_error:
                    print(f"Error releasing lock: {lock_error}")
