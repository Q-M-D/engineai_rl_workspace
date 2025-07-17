from engineai_rl.algos.ppo.config_ppo import ConfigPpo


class ConfigPpoContact(ConfigPpo):
    class networks(ConfigPpo.networks):
        # contact network is used during training but not optimized with PPO
        training = ["actor", "critic"]
        inference = ["actor"]

        class contact:
            class_name = "AttentionNetwork"
            input_infos = {"num_input_dim": "contact"}
            output_infos = {"num_output_dim": 4}
            hidden_dim = 128
            num_heads = 4
            num_layers = 2

    class input(ConfigPpo.input):
        training = ["actor", "critic", "contact"]
        inference = ["actor", "contact"]

        class components(ConfigPpo.input.components):
            class contact:
                obs_list = [
                    "dof_pos",
                    "dof_vel",
                    "base_ang_vel",
                    "base_euler_xyz",
                ]
                obs_with_goals = False
                obs_history_length = 1
                obs_goals_history = False
                obs_history_with_goals = False
