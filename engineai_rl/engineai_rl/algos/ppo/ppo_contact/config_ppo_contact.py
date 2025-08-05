from engineai_rl.algos.ppo.config_ppo import ConfigPpo


class ConfigPpoContact(ConfigPpo):
    class networks(ConfigPpo.networks):
        # contact network is used during training but not optimized with PPO
        training = ["actor", "critic"]
        inference = ["actor"]
        
        class actor:
            class_name = "Mlp"
            input_infos = {"num_input_dim": "actor"}
            output_infos = {"num_output_dim": "action"}
            forward_inputs = ["obs", "contact_estimation"]
            forward_input_dims_infos = {"obs": "num_input_dim", "contact_estimation": "num_contact_dim"}
            hidden_dims = [512, 256, 128]
            activation = "elu"
        
        class critic:
            class_name = "Mlp"
            input_infos = {"num_input_dim": "critic"}
            output_infos = {"num_output_dim": "value"}
            hidden_dims = [512, 256, 128]
            activation = "elu"

        class contact:
            class_name = "AttentionNetwork"
            input_infos = {"num_input_dim": "contact"}
            output_infos = {"num_output_dim": 8}
            num_heads = 1
            obs_history_length = 20 # This is the length of the history for contact observations

    class policy(ConfigPpo.policy):
        contact_network_learning_rate = 1e-3
    
    class input(ConfigPpo.input):
        training = ["actor", "critic", "contact"]
        inference = ["actor", "contact"]

        class components(ConfigPpo.input.components):
            class contact:
                obs_list = [
                    "dof_pos",
                    "dof_vel",
                    "actions",
                    "base_ang_vel",
                    "base_euler_xyz",
                ]
                obs_with_goals = False
                obs_history_length = 20
                obs_goals_history = False
                obs_history_with_goals = False
                lag = True
