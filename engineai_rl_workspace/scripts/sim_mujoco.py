import os, asyncio
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


async def play(args):
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

    checkout_commit_or_branch(repo, current_commit, current_branch)
    unstash_files(repo)
    if lock.redis.get(lock.lock_key) == lock.pid.encode():
        lock.release()
    print(INITIALIZATION_COMPLETE_MESSAGE)
    
    os._exit(0)



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


if __name__ == "__main__":
    global lock, repo, current_commit, current_branch
    try:
        MOVE_CAMERA = False
        args = get_args()
        if args.use_joystick:
            use_joystick(args)
        asyncio.run(play(args))
    except KeyboardInterrupt or SystemExit:
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            try:
                checkout_commit_or_branch(repo, current_commit, current_branch)
                unstash_files(repo)
            finally:
                lock.release()
