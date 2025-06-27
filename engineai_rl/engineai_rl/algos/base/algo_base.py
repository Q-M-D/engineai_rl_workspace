from __future__ import annotations

from abc import ABC


class AlgoBase(ABC):
    def __init__(
        self,
        networks,
        policy_cfg,
        env,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.networks = networks

    def eval_mode(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    @property
    def inference_policy(self):
        raise NotImplementedError

    def act(self, inputs):
        raise NotImplementedError

    def process_env_step(self, rewards, dones, infos, **kwargs):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
