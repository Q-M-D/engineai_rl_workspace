from engineai_rl_workspace import (
    REDIS_HOST,
    LOCK_KEY,
    REDIS_PORT,
    LOCK_TIMEOUT,
    LOCK_MESSAGE,
    PROGRAM_START_MESSAGE,
    INITIALIZATION_COMPLETE_MESSAGE,
    ENGINEAI_WORKSPACE_ROOT_DIR,
)

print(PROGRAM_START_MESSAGE)
import os, asyncio
from git import Repo

from engineai_gym.wrapper import VecGymWrapper, RecordVideoWrapper

from engineai_rl_workspace.utils import (
    get_args,
    generate_cfg_files_from_json,
    get_dict_from_cfg_before_modification,
)
from engineai_rl_workspace.utils.process_resume_files import (
    get_log_root_and_log_dir,
    checkout_resume_commit,
)
from engineai_rl_workspace.utils.convert_between_py_and_dict import (
    update_cfg_dict_from_args,
)
from engineai_rl_lib.git import (
    store_code_state,
    get_current_commit_and_branch,
    checkout_commit_or_branch,
    unstash_files,
)
from engineai_rl_lib.json import save_json_files
from engineai_rl_lib.redis_lock import RedisLock
import torch.distributed as dist

GPU_WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
IS_DISTRIBUTED = GPU_WORLD_SIZE > 1

# if not distributed training, set local and global rank to 0 and return
if not IS_DISTRIBUTED:
    GPU_LOCAL_RANK = 0
    GPU_GLOBAL_RANK = 0
else:
    dist.init_process_group(backend="nccl", init_method="env://")
    # get rank and world size
    GPU_LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    GPU_GLOBAL_RANK = int(os.getenv("RANK", "0"))


async def train(args):
    global lock, repo, current_commit, current_branch
    if GPU_GLOBAL_RANK == 0:
        lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
        if not await lock.acquire():
            print("Could not acquire lock, exiting...")
            return
    repo = Repo(ENGINEAI_WORKSPACE_ROOT_DIR)
    if args.resume or args.run_exist and GPU_GLOBAL_RANK == 0:
        current_commit, current_branch = get_current_commit_and_branch(repo)
        _, log_dir = get_log_root_and_log_dir(args)
        checkout_resume_commit(log_dir, repo)
        generate_cfg_files_from_json(args)
        import engineai_rl_workspace.exps

        if IS_DISTRIBUTED:
            dist.barrier()
    else:
        if IS_DISTRIBUTED:
            dist.barrier()
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
    if not args.resume and not args.debug and GPU_GLOBAL_RANK == 0:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        store_code_state(log_dir, repo)
        env_cfg_raw_dict = get_dict_from_cfg_before_modification(env_cfg)
        algo_cfg_raw_dict = get_dict_from_cfg_before_modification(algo_cfg)
        cfg = {"env_cfg": env_cfg_raw_dict, "algo_cfg": algo_cfg_raw_dict}
        update_cfg_dict_from_args(cfg, args)
        save_json_files(cfg, log_dir=log_dir, filename="config.json")
    if IS_DISTRIBUTED:
        dist.barrier()
    if args.resume or args.run_exist and GPU_GLOBAL_RANK == 0:
        checkout_commit_or_branch(repo, current_commit, current_branch)
        unstash_files(repo)
    if GPU_GLOBAL_RANK == 0:
        if lock.redis.get(lock.lock_key) == lock.pid.encode():
            lock.release()
    print(INITIALIZATION_COMPLETE_MESSAGE)
    if args.sim_devices:
        args.sim_device = args.sim_devices[GPU_LOCAL_RANK]
    if args.rl_devices:
        args.rl_device = args.rl_devices[GPU_LOCAL_RANK]
    env = exp_registry.make_env(
        task_class,
        obs_class,
        goal_class,
        domain_rand_class,
        reward_class,
        args,
        env_cfg,
    )
    env = VecGymWrapper(env)
    if args.video and not args.debug and GPU_GLOBAL_RANK == 0:
        env = RecordVideoWrapper(
            env,
            frame_size=args.frame_size,
            fps=args.fps,
            record_interval=args.record_interval,
            record_length=args.record_length,
            num_steps_per_env=algo_cfg.runner.num_steps_per_env,
            env_idx=args.env_idx_record,
            actor_idx=args.actor_idx_record,
            rigid_body_idx=args.rigid_body_idx_record,
            camera_offset=args.camera_offset,
            camera_rotation=args.camera_rotation,
            video_path=os.path.join(log_dir, "train_videos"),
        )
    ppo_runner = exp_registry.make_alg_runner(env, args.exp_name, args, log_dir)
    ppo_runner.learn(
        num_learning_iterations=algo_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    global lock, repo, current_commit, current_branch
    try:
        args = get_args()
        asyncio.run(train(args))
    except KeyboardInterrupt or SystemExit:
        try:
            if (
                GPU_GLOBAL_RANK == 0
                and lock.redis.get(lock.lock_key) == lock.pid.encode()
            ):
                checkout_commit_or_branch(repo, current_commit, current_branch)
                unstash_files(repo)
                lock.release()
        except:
            pass
