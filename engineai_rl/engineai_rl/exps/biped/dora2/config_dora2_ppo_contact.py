from engineai_rl.algos.ppo.ppo_contact.config_ppo_contact import ConfigPpoContact


class ConfigDora2PpoContact(ConfigPpoContact):
    class params(ConfigPpoContact.params):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9

    class runner(ConfigPpoContact.runner):
        seed = 5
        num_steps_per_env = 60
        max_iterations = 30000

    class networks(ConfigPpoContact.networks):
        class critic(ConfigPpoContact.networks.critic):
            hidden_dims = [768, 256, 128]

    class input(ConfigPpoContact.input):
        class components(ConfigPpoContact.input.components):
            goal_list = ["gait_phase", "commands"]

            class actor(ConfigPpoContact.input.components.actor):
                obs_goals_history = True
                obs_history_length =20
                obs_list = [
                    "dof_pos",          # 12
                    "dof_vel",          # 12
                    "actions",          # 12
                    "base_ang_vel",     # 3
                    "base_euler_xyz",   # 3
                    "contact_mask",     # 2
                ]

            class critic(ConfigPpoContact.input.components.critic):
                obs_goals_history = True
                obs_history_length = 3
                obs_list = [
                    "base_lin_vel",
                    "dof_pos",
                    "dof_vel",
                    "actions",
                    "dof_pos_ref_diff",
                    "base_ang_vel",
                    "base_euler_xyz",
                    "rand_push_force",
                    "rand_push_torque",
                    "terrain_frictions",
                    "body_mass",
                    "stance_curve",
                    "swing_curve",
                    "contact_mask",
                    "height_measurements",
                ]

        class obs_noise(ConfigPpoContact.input.obs_noise):
            noise_level = 1.5

            class scales(ConfigPpoContact.input.obs_noise.scales):
                actor = {
                    "base_ang_vel": 0.2,
                    "projected_gravity": 0.05,
                    "dof_pos": 0.02,
                    "dof_vel": 2.5,
                    "base_euler_xyz": 0.1,
                }

