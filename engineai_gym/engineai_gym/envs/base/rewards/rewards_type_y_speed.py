from engineai_gym.envs.base.rewards.rewards_base import RewardsBase
import torch


class RewardsTypeBaseVel(RewardsBase):
    def reward_low_speed_y(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        This specific implementation focuses on the y-axis speed.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.env.base_lin_vel[:, 1])
        absolute_command = torch.abs(self.env.commands[:, 1])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.env.base_lin_vel[:, 1]) != torch.sign(
            self.env.commands[:, 1]
        )

        # Initialize reward tensor
        reward = torch.zeros_like(self.env.base_lin_vel[:, 1])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 2.0
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.env.commands[:, 1].abs() > 0.1)
