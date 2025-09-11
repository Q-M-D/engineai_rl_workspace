from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch
from engineai_gym.envs import BipedRobot


class N2(BipedRobot):
    def compute_ref_state(self):
        phase = self.get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.params.target_joint_pos_scale
        scale_2 = 2 * scale_1
        
        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 6] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 7] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 8] = sin_pos_l * scale_1
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 15] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 16] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 17] = -sin_pos_r * scale_1

        self.ref_dof_pos[torch.abs(sin_pos) < 0.05] = 0.0

        self.ref_action = 2 * self.ref_dof_pos

        self.ref_dof_pos += self.default_dof_pos
