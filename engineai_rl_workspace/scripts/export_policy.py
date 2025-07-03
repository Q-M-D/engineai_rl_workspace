import os, asyncio
from collections import OrderedDict
from git import Repo

from engineai_rl_workspace.utils import (
    get_args,
    get_load_run_path,
    get_load_checkpoint_path,
    generate_cfg_files_from_json,
    convert_nn_to_onnx,
    convert_onnx_to_mnn,
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
from engineai_rl_lib.git import (
    get_current_commit_and_branch,
    checkout_commit_or_branch,
    unstash_files,
)
from engineai_rl_lib.class_operations import class_to_dict
from engineai_rl_lib.redis_lock import RedisLock
from engineai_rl_lib.networks import CombinedNetworks
from engineai_rl.modules.networks import *

import torch


async def export_policy(args):
    global lock, repo, current_commit, current_branch
    lock = RedisLock(REDIS_HOST, LOCK_KEY, REDIS_PORT, LOCK_TIMEOUT, LOCK_MESSAGE)
    if not await lock.acquire():
        print("Could not acquire lock, exiting...")
        return
    args.resume = True
    repo = Repo(ENGINEAI_WORKSPACE_ROOT_DIR)
    current_commit, current_branch = get_current_commit_and_branch(repo)
    _, log_dir = get_log_root_and_log_dir(args)
    checkout_resume_commit(log_dir, repo)
    generate_cfg_files_from_json(args)
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
    log_dir, load_run = get_load_run_path(log_root, load_run=args.load_run)
    load_checkpoint = get_load_checkpoint_path(
        load_run=log_dir, checkpoint=args.checkpoint
    )
    path = os.path.join(log_dir, "policies")
    loaded_dict = torch.load(load_checkpoint, map_location=torch.device(args.rl_device))
    inference_network_names = algo_cfg.networks.inference
    input_sizes = loaded_dict["infos"]["input_sizes"]
    inference_networks = OrderedDict()
    for inference_network_name in inference_network_names:
        inference_network_cfg = class_to_dict(
            eval(f"algo_cfg.networks.{inference_network_name}")
        )
        inference_network_class_name = inference_network_cfg.pop("class_name")
        inference_network_class = eval(inference_network_class_name)
        network_input_infos = inference_network_cfg.pop("input_infos")
        input_dim_infos = {}
        for network_input_name, network_input_type in network_input_infos.items():
            if isinstance(network_input_type, list):
                input_dim_infos[network_input_name] = 0
                for network_input_subtype in network_input_type:
                    if isinstance(network_input_subtype, int):
                        input_dim_infos[network_input_name] += network_input_subtype
                    elif network_input_subtype in input_sizes:
                        input_dim_infos[network_input_name] += input_sizes[
                            network_input_subtype
                        ]
                    else:
                        raise ValueError(
                            f"Network input type {network_input_subtype} not supported"
                        )
            else:
                if isinstance(network_input_type, int):
                    input_dim_infos[network_input_name] = network_input_type
                elif network_input_type in input_sizes:
                    input_dim_infos[network_input_name] = input_sizes[
                        network_input_type
                    ]
                else:
                    raise ValueError(
                        f"Network input type {network_input_type} not supported"
                    )
        network_output_infos = inference_network_cfg.pop("output_infos")
        output_dim_infos = {}
        for (
            network_output_name,
            network_output_type,
        ) in network_output_infos.items():
            if isinstance(network_output_type, list):
                input_dim_infos[network_output_name] = 0
                for network_output_subtype in network_output_type:
                    if network_output_subtype == "action":
                        output_dim_infos[network_output_name] += len(
                            env_cfg.env.action_joints
                        )
                    elif network_output_subtype == "value":
                        output_dim_infos[network_output_name] += 1
                    elif isinstance(network_output_subtype, int):
                        output_dim_infos[network_output_name] += network_output_subtype
                    else:
                        raise ValueError(
                            f"Network output type {network_output_subtype} not supported"
                        )
            else:
                if network_output_type == "action":
                    output_dim_infos[network_output_name] = len(
                        env_cfg.env.action_joints
                    )
                elif network_output_type == "value":
                    output_dim_infos[network_output_name] = 1
                elif isinstance(network_output_type, int):
                    output_dim_infos[network_output_name] = network_output_type
                else:
                    raise ValueError(
                        f"Network output type {network_output_type} not supported"
                    )
        if "forward_inputs" in inference_network_cfg:
            forward_inputs = inference_network_cfg.pop("forward_inputs")
        else:
            forward_inputs = None
        if "forward_input_dims_infos" in inference_network_cfg:
            forward_input_dims_infos = inference_network_cfg.pop(
                "forward_input_dims_infos"
            )
            forward_input_dims = {}
            for (
                forward_input_name,
                forward_input_type,
            ) in forward_input_dims_infos.items():
                if isinstance(forward_input_type, list):
                    forward_input_dims[forward_input_type] = 0
                    for forward_input_subtype in forward_input_type:
                        if isinstance(forward_input_subtype, int):
                            forward_input_dims[
                                forward_input_name
                            ] += forward_input_subtype
                        elif forward_input_subtype in input_dim_infos:
                            forward_input_dims[forward_input_name] += input_dim_infos[
                                forward_input_subtype
                            ]
                        else:
                            raise ValueError(
                                f"Forward input type {forward_input_subtype} not supported"
                            )
                else:
                    if isinstance(forward_input_type, int):
                        forward_input_dims[forward_input_name] = forward_input_type
                    elif forward_input_type in input_dim_infos:
                        forward_input_dims[forward_input_name] = input_dim_infos[
                            forward_input_type
                        ]
                    else:
                        raise ValueError(
                            f"Forward input type {forward_input_type} not supported"
                        )
        else:
            forward_input_dims = None

        if "forward_outputs" in inference_network_cfg:
            forward_outputs = inference_network_cfg.pop("forward_outputs")
        else:
            forward_outputs = None
        if "forward_output_dims_infos" in inference_network_cfg:
            forward_output_dims_infos = inference_network_cfg.pop(
                "forward_output_dims_infos"
            )
            forward_output_dims = {}
            for (
                forward_output_name,
                forward_output_type,
            ) in forward_output_dims_infos.items():
                if isinstance(forward_output_type, list):
                    forward_output_dims[forward_output_name] = 0
                    for forward_output_subtype in forward_output_type:
                        if forward_output_subtype in network_output_infos:
                            forward_output_dims[
                                forward_output_name
                            ] += output_dim_infos[forward_output_subtype]
                        elif isinstance(forward_output_subtype, int):
                            forward_output_dims[
                                forward_output_name
                            ] += forward_output_subtype
                        else:
                            raise ValueError(
                                f"Forward output type {forward_output_subtype} not supported"
                            )
                else:
                    if forward_output_type in network_output_infos:
                        forward_output_dims[forward_output_name] = output_dim_infos[
                            forward_output_type
                        ]
                    elif isinstance(forward_output_type, int):
                        forward_output_dims[forward_output_name] = forward_output_type
                    else:
                        raise ValueError(
                            f"Forward output type {forward_output_type} not supported"
                        )
        else:
            forward_output_dims = None

        if inference_network_cfg.get("normalizer_class_name", False):
            normalizer_class = eval(inference_network_cfg.pop("normalizer_class_name"))
            normalizer = normalizer_class(
                **input_dim_infos,
                **inference_network_cfg.pop("normalizer_args"),
            )
        else:
            normalizer = None
        inference_network = inference_network_class(
            **input_dim_infos,
            **output_dim_infos,
            **inference_network_cfg,
            normalizer=normalizer,
        ).to("cpu")
        inference_network.load_state_dict(
            loaded_dict["model_state_dict"][inference_network_name]
        )
        inference_networks[inference_network_name] = {
            "network": inference_network,
            "input_dim_infos": input_dim_infos,
            "output_dim_infos": output_dim_infos,
            "forward_input_dims": forward_input_dims,
            "forward_inputs": forward_inputs,
            "forward_output_dims": forward_output_dims,
            "forward_outputs": forward_outputs,
        }
    network_list = [
        inference_network for inference_network in inference_networks.values()
    ]
    combined_networks = CombinedNetworks(network_list)

    convert_nn_to_onnx(
        combined_networks, path, args.exp_name + "_" + args.load_run + "_policy"
    )

    convert_onnx_to_mnn(
        os.path.join(
            path,
            args.exp_name + "_" + args.load_run + "_policy" + ".onnx",
        ),
        os.path.join(
            path,
            args.exp_name + "_" + args.load_run + "_policy" + ".mnn",
        ),
    )


if __name__ == "__main__":
    global lock, repo, current_commit, current_branch
    try:
        args = get_args()
        asyncio.run(export_policy(args))
    except KeyboardInterrupt or SystemExit:
        try:
            if lock.redis.get(lock.lock_key) == lock.pid.encode():
                checkout_commit_or_branch(repo, current_commit, current_branch)
                unstash_files(repo)
                lock.release()
        except:
            pass
