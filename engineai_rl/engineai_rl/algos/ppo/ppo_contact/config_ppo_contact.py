from engineai_rl.algos.ppo.config_ppo import ConfigPpo


class ConfigPpoContact(ConfigPpo):
    class networks(ConfigPpo.networks):
        # contact network is used during training but not optimized with PPO
        training = ["actor", "critic", "contact"] # not use in runner_base.py:82
        inference = ["actor", "contact"]
        
        class actor:
            class_name = "Mlp"
            input_infos = {"num_input_dim": "actor"}
            output_infos = {"num_output_dim": "action"}
            forward_inputs = ["obs"]
            forward_input_dims_infos = {"obs": "num_input_dim"}
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
            output_infos = {"num_output_dim": 2}
            forward_inputs = ["obs"]
            forward_input_dims_infos = {"obs": "num_input_dim"}
            num_heads = 1
            obs_history_length = 20 # This is the length of the history for contact observations

    class policy(ConfigPpo.policy):
        contact_network_learning_rate = 1e-3
    
    class input(ConfigPpo.input):
        training = ["actor", "critic", "contact", "contact_ground_truth"]
        inference = ["actor", "contact"]

        class components(ConfigPpo.input.components):
            class contact:
                obs_list = [
                    "dof_pos",          # 12
                    "dof_vel",          # 12
                    "actions",          # 12
                    "base_ang_vel",     # 3
                    "base_euler_xyz",   # 3
                ]
                # when obs_with_goals = False and obs_history_length <= 1, the input is obs
                # when obs_with_goals = True and obs_history_length <= 1, the input is torch.cat([obs, goals], dim=-1)
                # when obs_goals_history = True and obs_history_length > 1, the input is torch.cat(history([obs, goals]), dim=-1)
                # when obs_goals_history = False,  obs_history_with_goals = True, and obs_history_length > 1, the input is torch.cat([history(obs), goals], dim=-1)
                # when obs_goals_history = False,  obs_history_with_goals = False, and obs_history_length > 1, the input is history(obs)
                obs_with_goals = False
                obs_history_length = 20
                obs_goals_history = False
                obs_history_with_goals = False
                lag = True
                
                        
            class contact_ground_truth():
                obs_list = [
                    "contact_mask"
                ]
                obs_with_goals = False
                obs_history_length = 1
                obs_goals_history = False
                obs_history_with_goals = False
                # retrieve obs before reset for terminal state
                obs_before_reset = False
                # add obs_lag in domain_rands
                lag = True
