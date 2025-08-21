from collections import deque
import torch
from engineai_rl_lib.device import input_to_device
from engineai_rl_lib.class_operations import (
    add_instance_properties_and_methods_to_class,
)


class InputRetrivalEnvWrapper:
    def __init__(self, env, obs_cfg, device="cpu"):
        self._obs_cfg = obs_cfg
        self._env = env
        add_instance_properties_and_methods_to_class(env, self)
        delattr(type(self), "device")
        setattr(type(self), "device", property(lambda self: device))
        self._input_sizes = self.init_input()
        self.init_obs()

    def reset(self, contact_net, set_goals_callback=None, set_goals_callback_args=None):
        _, _, infos = self._env.reset()
        if set_goals_callback is not None:
            set_goals_callback(*set_goals_callback_args)
        obs_dict, goal_dict, infos = self._env.get_env_info()
        obs_dict, goal_dict = input_to_device(obs_dict, self.device), input_to_device(
            goal_dict, self.device
        )
        obs = self.retrieve_obs(obs_dict)
        goals = self.retrieve_goals(goal_dict)
        if self._obs_cfg["obs_noise"]["add_noise"]:
            self.get_noise_scale_vec(obs_dict, obs)
            self.add_obs_noise(obs)
        for obs_type, obs_subtypes in self._obs_types.items():
            if obs_subtypes:
                for obs_subtype in obs_subtypes:
                    obs[obs_type][obs_subtype] = torch.clip(
                        obs[obs_type][obs_subtype],
                        -self._obs_cfg["obs_clip_threshold"],
                        self._obs_cfg["obs_clip_threshold"],
                    )
            else:
                obs[obs_type] = torch.clip(
                    obs[obs_type],
                    -self._obs_cfg["obs_clip_threshold"],
                    self._obs_cfg["obs_clip_threshold"],
                )
        
        # contact estimator part
        predict_contact, weight = self.predict_contact(
            obs['contact'], 
            contact_net, 
            torch.ones(self._env.num_envs, dtype=torch.bool, device=self.device),
            goals
            ) if 'contact' in obs else (None, None)
        obs = self.update_obs_with_contact(obs, obs_dict, predict_contact)
        
        inputs = self.get_inputs(
            torch.ones(self._env.num_envs, dtype=torch.bool, device=self.device),
            goals,
            obs,
        )
        inputs = input_to_device(inputs, self.device)
        return inputs

    def step(self, actions, contact_net=None, set_goals_callback=None, set_goals_callback_args=None):
        obs_dict, goal_dict, rewards, dones, infos = self._env.step(actions)
        if set_goals_callback is not None:
            goal_dict = set_goals_callback(*set_goals_callback_args)
        obs_dict, goal_dict = input_to_device(obs_dict, self.device), input_to_device(
            goal_dict, self.device
        )
        obs = self.retrieve_obs(obs_dict)
        goals = self.retrieve_goals(goal_dict)
        if self._obs_cfg["obs_noise"]["add_noise"]:
            self.add_obs_noise(obs)
        for obs_type, obs_subtypes in self._obs_types.items():
            if obs_subtypes:
                for obs_subtype in obs_subtypes:
                    obs[obs_type][obs_subtype] = torch.clip(
                        obs[obs_type][obs_subtype],
                        -self._obs_cfg["obs_clip_threshold"],
                        self._obs_cfg["obs_clip_threshold"],
                    )
            else:
                obs[obs_type] = torch.clip(
                    obs[obs_type],
                    -self._obs_cfg["obs_clip_threshold"],
                    self._obs_cfg["obs_clip_threshold"],
                )
        
        # contact estimator part
        predict_contact, weight = self.predict_contact(
            obs['contact'], 
            contact_net, 
            torch.ones(self._env.num_envs, dtype=torch.bool, device=self.device),
            goals
            ) if 'contact' in obs else (None, None)
        obs = self.update_obs_with_contact(obs, obs_dict, predict_contact)
        
        next_inputs = self.get_inputs(dones, goals, obs)
        next_inputs, rewards, dones = (
            input_to_device(next_inputs, self.device),
            input_to_device(rewards, self.device),
            input_to_device(dones, self.device),
        )
        return next_inputs, actions, dones, infos, rewards

    def predict_contact(self, contact_obs, contact_net, dones, goals):
        # 1. get stacked contact obs
        obs_history_tmp = self.obs_history.copy()
        obs_history_deque_tmp = self.obs_history_deque.copy()
        obs_goals_history_tmp = self.obs_goals_history.copy()
        obs_goals_history_deque_tmp = self.obs_goals_history_deque.copy()
        
        # ============
        # preprocess data
        # ============
        if not hasattr(self, "obs_goals_history"):
            if "contact" in obs_history_tmp:
                for i in range(obs_history_deque_tmp["contact"].maxlen):
                    obs_history_deque_tmp["contact"][i][dones] = 0
        else:
            if "contact" in obs_goals_history_tmp:
                for i in range(obs_history_deque_tmp["contact"].maxlen):
                    obs_goals_history_deque_tmp["contact"][i][dones] = 0
        if hasattr(self, "obs_goals_history") and "contact" in obs_goals_history_deque_tmp:
            obs_goals_history_deque_tmp["contact"].append(
                torch.cat((contact_obs, goals), dim=1)
            )
            obs_goals_history_tmp["contact"] = torch.cat(
                [
                    obs_goals_history_deque_tmp["contact"][i]
                    for i in range(
                        obs_goals_history_deque_tmp["contact"].maxlen
                    )
                ],
                dim=1,
            )
        else:
            if "contact" in obs_history_deque_tmp:
                obs_history_deque_tmp["contact"].append(contact_obs)
                obs_history_tmp["contact"] = torch.cat(
                    [
                        obs_history_deque_tmp["contact"][i]
                        for i in range(obs_history_deque_tmp["contact"].maxlen)
                    ],
                    dim=1,
                )
        
        # ============
        # process data
        # ============
        if (
            self._obs_cfg["components"]["contact"].get("obs_history_length", 1)
            > 1
        ):
            if self._obs_cfg["components"]["contact"]["obs_goals_history"]:
                contact_net_input = obs_goals_history_tmp["contact"]
            elif self._obs_cfg["components"]["contact"][
                "obs_history_with_goals"
            ]:
                contact_net_input = torch.cat(
                    (obs_history_tmp["contact"], goals), dim=-1
                )
            else:
                contact_net_input = obs_history_tmp["contact"]
        else:
            if self._obs_cfg["components"]["contact"]["obs_with_goals"]:
                contact_net_input = torch.cat((contact_obs, goals), dim=-1)
            else:
                contact_net_input = contact_obs

        # 2. predict contact using contact_net and return
        return contact_net(contact_net_input)

    def update_obs_with_contact(self, obs, obs_dict, predict_contact):
        if not self._obs_cfg["components"]["contact"]["obs_list"]:
            obs["contact"] = torch.empty(
                self._env.num_envs, 0, device=self.device
            )
        else:
            if self._obs_cfg["components"]["contact"]["lag"]:
                obs_list = []
                for obs_name in self._obs_cfg["components"]["contact"][
                    "obs_list"
                ]:
                    if obs_name == "contact_mask":
                        obs_dict.append(predict_contact)
                        continue
                    if obs_name in obs_dict["lagged_obs"]:
                        obs_list.append(
                            obs_dict["lagged_obs"]["after_reset"][obs_name]
                        )
                    else:
                        obs_list.append(
                            obs_dict["non_lagged_obs"]["after_reset"][obs_name]
                        )
            else:
                obs_list = [
                    predict_contact if obs_name == "contact_mask" else
                    obs_dict["non_lagged_obs"]["after_reset"][obs_name]
                    for obs_name in self._obs_cfg["components"]["contact"][
                        "obs_list"
                    ]
                ]
            obs["contact"] = torch.cat(obs_list, dim=-1)
        return obs

    def init_obs(self):
        stacked_obs = False
        for obs_type in self._obs_types:
            if self._obs_cfg["components"][obs_type].get("obs_history_length", 1) > 1:
                stacked_obs = True
                break
        if stacked_obs:
            self.init_stacked_obs()

    def init_input(self):
        self._obs_types = self.get_obs_types()
        obs_dict, goal_dict, _ = self._env.reset()
        # remove height_measurements when it's not in obs_dict so algo_cfg doesn't need to distinguish rough and flat
        if "height_measurements" not in obs_dict["non_lagged_obs"]["after_reset"]:
            for obs_type in self._obs_types:
                if (
                    "height_measurements"
                    in self._obs_cfg["components"][obs_type]["obs_list"]
                ):
                    self._obs_cfg["components"][obs_type]["obs_list"].remove(
                        "height_measurements"
                    )
        self.obs_sizes = self.get_obs_sizes(obs_dict)
        self.goal_sizes = self.get_goal_sizes(goal_dict)
        input_sizes = self.get_input_sizes()
        return input_sizes

    def get_inputs(self, dones, goals, obs):
        if hasattr(self, "obs_history"):
            self.get_stacked_obs(dones, goals, obs)
        inputs = {}
        for obs_type, _ in self._obs_types.items():
            if (
                self._obs_cfg["components"][obs_type].get("obs_history_length", 1)
                > 1
            ):
                if self._obs_cfg["components"][obs_type]["obs_goals_history"]:
                    inputs[obs_type] = self.obs_goals_history[obs_type]
                elif self._obs_cfg["components"][obs_type][
                    "obs_history_with_goals"
                ]:
                    inputs[obs_type] = torch.cat(
                        (self.obs_history[obs_type], goals), dim=-1
                    )
                else:
                    inputs[obs_type] = self.obs_history[obs_type]
            else:
                if self._obs_cfg["components"][obs_type]["obs_with_goals"]:
                    inputs[obs_type] = torch.cat((obs[obs_type], goals), dim=-1)
                else:
                    inputs[obs_type] = obs[obs_type]
        return inputs

    def init_stacked_obs(self):
        self.obs_history_deque = {}
        self.obs_history = {}
        for obs_type, _ in self._obs_types.items():
            if self._obs_cfg["components"][obs_type].get("obs_history_length", 1) > 1:
                self.obs_history_deque[obs_type] = deque(
                    maxlen=self.obs_cfg["components"][obs_type][
                        "obs_history_length"
                    ]
                )
                for _ in range(
                    self.obs_cfg["components"][obs_type]["obs_history_length"]
                ):
                    self.obs_history_deque[obs_type].append(
                        torch.zeros(
                            self._env.num_envs,
                            self.obs_sizes[obs_type],
                            dtype=torch.float,
                            device=self.device,
                        )
                    )
        obs_goals_history = False
        for obs_type in self._obs_types:
            if self._obs_cfg["components"][obs_type].get("obs_history_length", 1) > 1:
                if self._obs_cfg["components"][obs_type]["obs_goals_history"]:
                    obs_goals_history = True
                    break
        if obs_goals_history:
            self.obs_goals_history_deque = {}
            self.obs_goals_history = {}
            for obs_type, _ in self._obs_types.items():
                if (
                    self._obs_cfg["components"][obs_type]["obs_goals_history"]
                    and self._obs_cfg["components"][obs_type].get(
                        "obs_history_length", 1
                    )
                    > 1
                ):
                    self.obs_goals_history_deque[obs_type] = deque(
                        maxlen=self.obs_cfg["components"][obs_type][
                            "obs_history_length"
                        ]
                    )
                    for _ in range(
                        self.obs_cfg["components"][obs_type]["obs_history_length"]
                    ):
                        self.obs_goals_history_deque[obs_type].append(
                            torch.zeros(
                                self._env.num_envs,
                                self.obs_sizes[obs_type]
                                + self.goal_sizes[obs_type],
                                dtype=torch.float,
                                device=self.device,
                            )
                        )

    def get_stacked_obs(self, dones, goals, obs):
        self.reset_stacked_obs(dones)
        for obs_type, _ in self._obs_types.items():
            if obs_type in self.obs_history_deque:
                    self.obs_history_deque[obs_type].append(obs[obs_type])
                    self.obs_history[obs_type] = torch.cat(
                        [
                            self.obs_history_deque[obs_type][i]
                            for i in range(self.obs_history_deque[obs_type].maxlen)
                        ],
                        dim=1,
                    )
            if obs_type in self.obs_goals_history_deque:
                    self.obs_goals_history_deque[obs_type].append(
                        torch.cat((obs[obs_type], goals), dim=1)
                    )
                    self.obs_goals_history[obs_type] = torch.cat(
                        [
                            self.obs_goals_history_deque[obs_type][i]
                            for i in range(
                                self.obs_goals_history_deque[obs_type].maxlen
                            )
                        ],
                        dim=1,
                    )

    def reset_stacked_obs(self, dones):
        for obs_type, _ in self._obs_types.items():
            if obs_type in self.obs_history_deque:
                    for i in range(self.obs_history_deque[obs_type].maxlen):
                        self.obs_history_deque[obs_type][i][dones] = 0
            if obs_type in self.obs_goals_history_deque:
                for i in range(self.obs_history_deque[obs_type].maxlen):
                    self.obs_goals_history_deque[obs_type][i][dones] = 0

    def retrieve_obs(self, obs_dict):
        obs = {}
        for obs_type, _ in self._obs_types.items():
            if not self._obs_cfg["components"][obs_type]["obs_list"]:
                obs[obs_type] = torch.empty(
                    self._env.num_envs, 0, device=self.device
                )
            else:
                if self._obs_cfg["components"][obs_type]["lag"]:
                    obs_list = []
                    for obs_name in self._obs_cfg["components"][obs_type][
                        "obs_list"
                    ]:
                        if obs_name in obs_dict["lagged_obs"]:
                            obs_list.append(
                                obs_dict["lagged_obs"]["after_reset"][obs_name]
                            )
                        else:
                            obs_list.append(
                                obs_dict["non_lagged_obs"]["after_reset"][obs_name]
                            )
                else:
                    obs_list = [
                        obs_dict["non_lagged_obs"]["after_reset"][obs_name]
                        for obs_name in self._obs_cfg["components"][obs_type][
                            "obs_list"
                        ]
                    ]
                obs[obs_type] = torch.cat(obs_list, dim=-1)
        return obs

    def retrieve_goals(self, goal_dict):
        goal_list = self._obs_cfg["components"].get("goal_list", [])
        if not goal_list:
            return torch.empty(self._env.num_envs, 0, device=self.device)

        return torch.cat(
            [goal_dict[goal_name] for goal_name in goal_list],
            dim=-1,
        )

    def add_obs_noise(self, obs):
        for obs_type, noise_vec in self.obs_noise_vecs.items():
            obs_subtypes = self._obs_types[obs_type]
            if obs_subtypes:
                for obs_subtype in obs_subtypes:
                    obs[obs_type][obs_subtype] += (
                        2 * torch.rand_like(obs[obs_type][obs_subtypes[0]]) - 1
                    ) * self.obs_noise_vecs[obs_type]
            else:
                obs[obs_type] += (
                    2 * torch.rand_like(obs[obs_type]) - 1
                ) * self.obs_noise_vecs[obs_type]

    def get_noise_scale_vec(self, obs_dict, obs):
        self.obs_noise_vecs = {}
        for obs_type, obs_scale in self._obs_cfg["obs_noise"]["scales"].items():
            obs_subtypes = self._obs_types[obs_type]
            if obs_subtypes:
                noise_vec = torch.zeros_like(obs[obs_type][obs_subtypes[0]])
            else:
                noise_vec = torch.zeros_like(obs[obs_type])
            noise_scales = {}
            for obs_type, noise_dict in self._obs_cfg["obs_noise"]["scales"].items():
                noise_scales[obs_type] = {}
                for obs_name in self._env.cfg.env.obs_list:
                    if isinstance(noise_dict.get(obs_name, 1), dict):
                        noise_scales_tensor = torch.zeros(
                            len(self._env.dof_names),
                            device=self.device,
                            dtype=torch.float,
                        )
                        for idx, joint_name in enumerate(self._env.dof_names):
                            for (joint_type, obs_scale) in noise_dict.get(
                                obs_name
                            ).items():
                                if joint_type in joint_name:
                                    noise_scales_tensor[idx] = obs_scale
                        noise_scales[obs_type][obs_name] = noise_scales_tensor
                    else:
                        noise_scales[obs_type][obs_name] = noise_dict.get(obs_name, 1)
            noise_level = self._obs_cfg["obs_noise"]["noise_level"]
            idx = 0
            obs_list = self._obs_cfg["components"][obs_type]["obs_list"]
            for obs_name in obs_list:
                size = obs_dict["non_lagged_obs"]["after_reset"][obs_name].shape[1]
                noise_vec[:, idx : idx + size] = (
                    noise_scales[obs_type].get(obs_name, 0)
                    * noise_level
                    * self._env.obs_scales.get(obs_name, 1)
                )
                idx += size
            self.obs_noise_vecs[obs_type] = noise_vec

    def get_obs_sizes(self, obs_dict):
        obs_sizes = {}
        for obs_type in self._obs_types:
            obs_sizes[obs_type] = 0
            for obs_name in self._obs_cfg["components"][obs_type]["obs_list"]:
                if (
                    obs_name == "height_measurements"
                    and obs_name not in obs_dict["non_lagged_obs"]["after_reset"]
                ):
                    continue
                obs_sizes[obs_type] += obs_dict["non_lagged_obs"]["after_reset"][
                    obs_name
                ].shape[1]
        return obs_sizes

    def get_goal_sizes(self, goal_dict):
        goal_sizes = {}
        for obs_type in self._obs_types:
            goal_sizes[obs_type] = 0
            for goal_name in self._obs_cfg["components"]["goal_list"]:
                goal_sizes[obs_type] += goal_dict[goal_name].shape[1]
        return goal_sizes

    def get_input_sizes(self):
        input_sizes = {}
        for obs_type in self._obs_types:
            if self._obs_cfg["components"][obs_type].get("obs_history_length", 1) > 1:
                if self._obs_cfg["components"][obs_type]["obs_goals_history"]:
                    input_sizes[obs_type] = (
                        self.obs_sizes[obs_type] + self.goal_sizes[obs_type]
                    ) * self._obs_cfg["components"][obs_type]["obs_history_length"]
                elif self._obs_cfg["components"][obs_type]["obs_history_with_goals"]:
                    input_sizes[obs_type] = (
                        self.obs_sizes[obs_type]
                        * self._obs_cfg["components"][obs_type]["obs_history_length"]
                        + self.goal_sizes[obs_type]
                    )
                else:
                    input_sizes[obs_type] = (
                        self.obs_sizes[obs_type]
                        * self._obs_cfg["components"][obs_type]["obs_history_length"]
                    )
            else:
                if self._obs_cfg["components"][obs_type]["obs_with_goals"]:
                    input_sizes[obs_type] = (
                        self.obs_sizes[obs_type] + self.goal_sizes[obs_type]
                    )
                else:
                    input_sizes[obs_type] = self.obs_sizes[obs_type]

        return input_sizes

    def get_obs_types(self):
        obs_types = {}
        for obs_type in list(
            set(self._obs_cfg["training"] + self._obs_cfg["inference"])
        ):
            if self._obs_cfg["components"][obs_type].get("obs_before_reset", False):
                obs_types[obs_type] = ["before_reset", "after_reset"]
            else:
                obs_types[obs_type] = []
        return obs_types

    @property
    def obs_cfg(self):
        return self._obs_cfg

    @property
    def input_sizes(self):
        return self._input_sizes

    @property
    def obs_types(self):
        return self._obs_types
