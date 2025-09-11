from engineai_gym.envs.base.rewards.rewards import Rewards
from engineai_gym.envs.robots.biped.rewards_biped import RewardsBiped



import torch


class RewardN2(RewardsBiped):
    def reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.env.dof_pos - self.env.default_dof_pos
        left_yaw_roll = joint_diff[:, 4:6]
        right_yaw_roll = joint_diff[:,13:15]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)