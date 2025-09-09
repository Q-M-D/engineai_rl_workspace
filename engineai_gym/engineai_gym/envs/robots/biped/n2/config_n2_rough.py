from engineai_gym.envs.robots.biped.config_biped_robot import ConfigBipedRobot


class ConfigN2Rough(ConfigBipedRobot):
    class env(ConfigBipedRobot.env):
        num_envs = 2048
        action_joints = [
            "L_arm_shoulder_pitch_joint",
            "L_arm_shoulder_roll_joint",
            "L_arm_shoulder_yaw_joint",
            "L_arm_elbow_joint",
            "L_leg_hip_yaw_joint",
            "L_leg_hip_roll_joint",
            "L_leg_hip_pitch_joint",
            "L_leg_knee_joint",
            "L_leg_ankle_joint",
            
            "R_arm_shoulder_pitch_joint",
            "R_arm_shoulder_roll_joint",
            "R_arm_shoulder_yaw_joint",
            "R_arm_elbow_joint",
            "R_leg_hip_yaw_joint",
            "R_leg_hip_roll_joint",
            "R_leg_hip_pitch_joint",
            "R_leg_knee_joint",
            "R_leg_ankle_joint",
        ]
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
            "commands"
        ]
        goal_list = ["gait_phase", "commands"]
        use_ref_actions = False
        episode_length_s = 24

    class terrain(ConfigBipedRobot.terrain):
        static_friction = 0.6
        dynamic_friction = 0.6
        num_rows = 20  # number of terrain rows (levels)
        max_init_terrain_level = 10
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    class init_state(ConfigBipedRobot.init_state):
        pos = [0.0, 0.0, 0.68]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "L_arm_shoulder_pitch_joint": 0.0,
            "L_arm_shoulder_roll_joint": 0.3,
            "L_arm_shoulder_yaw_joint": 0.0,
            "L_arm_elbow_joint": 0.0,
            "L_leg_hip_yaw_joint": 0.0,
            "L_leg_hip_roll_joint": 0.0,
            "L_leg_hip_pitch_joint": 0.0,
            "L_leg_knee_joint": 0.0,
            "L_leg_ankle_joint": 0.0,
            "R_arm_shoulder_pitch_joint": 0.0,
            "R_arm_shoulder_roll_joint": -0.3,
            "R_arm_shoulder_yaw_joint": 0.0,
            "R_arm_elbow_joint": 0.0,
            "R_leg_hip_yaw_joint": 0.0,
            "R_leg_hip_roll_joint": 0.0,
            "R_leg_hip_pitch_joint": 0.0,
            "R_leg_knee_joint": 0.0,
            "R_leg_ankle_joint": 0.0,
        }

    class control(ConfigBipedRobot.control):
        # PD Drive parameters:
        control_type = "P"

        stiffness = {
            "arm_shoulder_pitch_joint": 40,
            "arm_shoulder_roll_joint": 40,
            "arm_shoulder_yaw_joint": 10,
            "arm_elbow_joint": 10,
            
            "hip_yaw_joint": 80,
            "hip_roll_joint": 80,
            "hip_pitch_joint": 80,
            "knee_joint": 80,
            "ankle_joint": 10,
        }
        damping = {
            "arm_shoulder_pitch_joint": 2.0,
            "arm_shoulder_roll_joint": 2.0,
            "arm_shoulder_yaw_joint": 0.5,
            "arm_elbow_joint": 0.5,

            "hip_roll_joint": 4.0,
            "hip_yaw_joint": 4.0,
            "hip_pitch_joint": 4.0,
            "knee_joint": 4.0,
            "ankle_joint": 0.5,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scales = {"joint": 0.5}
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class domain_rands(ConfigBipedRobot.domain_rands):
        class rigid_shape(ConfigBipedRobot.domain_rands.rigid_shape):
            randomize_friction = True
            friction_range = [0.2, 1.3]
            randomize_restitution = True
            restitution_range = [0.0, 0.4]

        class rigid_body(ConfigBipedRobot.domain_rands.rigid_body):
            randomize_base_mass = True
            added_mass_range = [-4.0, 4.0]

            randomize_com = True
            com_displacement_range = [-0.06, 0.06]

            randomize_link_mass = True
            link_mass_multi_range = [0.8, 1.2]

        class dof(ConfigBipedRobot.domain_rands.dof):
            randomize_gains = True
            stiffness_multi_range = [0.8, 1.2]
            damping_multi_range = [0.8, 1.2]

            randomize_torque = True
            torque_multi_range = [0.8, 1.2]

            randomize_motor_offset = True
            motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

            randomize_joint_friction = True
            randomize_joint_friction_each_joint = False
            joint_friction_multi_range = [0.01, 1.15]
            joint_friction_multi_range_each_joint = {
                "shoulder": [0.01, 1.15],
                "elbow": [0.01, 1.15],

                "hip": [0.01, 1.15],
                "knee": [0.01, 1.15],
                "ankle": [0.5, 1.3],
            }

            randomize_joint_armature = True
            randomize_joint_armature_each_joint = True
            joint_armature_multi_range = [0.27, 2]
            joint_armature_multi_range_each_joint = {
                "L_arm": [0.27, 2],
                "R_arm": [0.27, 2],

                "hip": [0.27, 2],
                "knee": [0.27, 2],
                "ankle": [0.27, 2],
            }

            randomize_coulomb_friction = False
            joint_coulomb_range = [0.1, 0.9]
            joint_viscous_range = [0.05, 0.1]

        class action_lag(ConfigBipedRobot.domain_rands.action_lag):
            action_lag_timesteps = 0
            randomize_action_lag_timesteps = True
            randomize_action_lag_timesteps_perstep = False
            action_lag_timesteps_range = [2, 5]

        class obs_lag(ConfigBipedRobot.domain_rands.obs_lag):
            motor_lag_timesteps = 0
            randomize_motor_lag_timesteps = True
            randomize_motor_lag_timesteps_perstep = False
            motor_lag_timesteps_range = [5, 15]
            imu_lag_timesteps = 0
            randomize_imu_lag_timesteps = False
            randomize_imu_lag_timesteps_perstep = False
            imu_lag_timesteps_range = [1, 10]

        class disturbance(ConfigBipedRobot.domain_rands.disturbance):
            push_robots = True
            push_interval_s = 8
            max_push_vel_xy = 0.4
            max_push_ang_vel = 0.6

    class asset(ConfigBipedRobot.asset):
        file = (
            "{ENGINEAI_GYM_PACKAGE_DIR}/resources/robots/biped/n2/urdf/N2.urdf"
        )

        name = "N2"
        foot_name = "ankle_link"
        knee_name = "knee_link"

        terminate_after_contacts_on = [
            "base_link",
            "l_arm_shoulder_pitch_Link",
            "l_arm_shoulder_roll_Link",
            "l_arm_shoulder_yaw_Link",
            "l_arm_elbow_Link",
            "l_arm_hand_Link",
            "r_arm_shoulder_pitch_Link",
            "r_arm_shoulder_roll_Link",
            "r_arm_shoulder_yaw_Link",
            "r_arm_elbow_Link",
            "r_arm_hand_Link",
            "l_leg_hip_yaw_link",
            "l_leg_hip_roll_link",
            "l_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_hip_yaw_link",
            "r_leg_hip_roll_link",
            "r_leg_hip_pitch_link",
            "r_leg_knee_link",
            ]
        penalize_contacts_on = ["base_link"]
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        joint_armature = {
            # Arm joints
            "L_arm_shoulder_pitch_joint": 0.01,
            "L_arm_shoulder_roll_joint": 0.01,
            "L_arm_shoulder_yaw_joint": 0.01,
            "L_arm_elbow_joint": 0.01,
            "R_arm_shoulder_pitch_joint": 0.01,
            "R_arm_shoulder_roll_joint": 0.01,
            "R_arm_shoulder_yaw_joint": 0.01,
            "R_arm_elbow_joint": 0.01,
            # Leg joints
            "L_leg_hip_yaw_joint": 0.01,
            "L_leg_hip_roll_joint": 0.01,
            "L_leg_hip_pitch_joint": 0.01,
            "L_leg_knee_joint": 0.01,
            "L_leg_ankle_joint": 0.01,
            "R_leg_hip_yaw_joint": 0.01,
            "R_leg_hip_roll_joint": 0.01,
            "R_leg_hip_pitch_joint": 0.01,
            "R_leg_knee_joint": 0.01,
            "R_leg_ankle_joint": 0.01,
        }
        joint_friction = {
            # Arm joints
            "L_arm_shoulder_pitch_joint": 0.0,
            "L_arm_shoulder_roll_joint": 0.0,
            "L_arm_shoulder_yaw_joint": 0.0,
            "L_arm_elbow_joint": 0.0,
            "R_arm_shoulder_pitch_joint": 0.0,
            "R_arm_shoulder_roll_joint": 0.0,
            "R_arm_shoulder_yaw_joint": 0.0,
            "R_arm_elbow_joint": 0.0,
            # Leg joints
            "L_leg_hip_yaw_joint": 0.0,
            "L_leg_hip_roll_joint": 0.0,
            "L_leg_hip_pitch_joint": 0.0,
            "L_leg_knee_joint": 0.0,
            "L_leg_ankle_joint": 0.0,
            "R_leg_hip_yaw_joint": 0.0,
            "R_leg_hip_roll_joint": 0.0,
            "R_leg_hip_pitch_joint": 0.0,
            "R_leg_knee_joint": 0.0,
            "R_leg_ankle_joint": 0.0,
        }

    class commands(ConfigBipedRobot.commands):
        curriculum = True
        max_curriculum = 1.7
        resampling_time = 8.0  # time before command are changed[s]
        yaw_from_heading_target = (
            False  # if true: compute ang vel command from heading error
        )
        num_commands = 3
        still_ratio = 0

        class ranges(ConfigBipedRobot.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]

    class normalization(ConfigBipedRobot.normalization):
        obs_scales = {
            "base_lin_vel": 2.0,
            "base_ang_vel": 1.0,
            "body_mass": 0.1,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "base_euler_xyz": 1.0,
            "height_measurements": 5.0,
        }

    class rewards(ConfigBipedRobot.rewards):
        class params(ConfigBipedRobot.rewards.params):
            base_height_target = 0.65
            max_contact_force = 500.0
            tracking_sigma = 5
            target_joint_pos_scale = 0.5
            target_feet_height = 0.1
            soft_dof_torque_limit_multi = {"joint": 0.9}

        class scales(ConfigBipedRobot.rewards.scales):
            torques = -1e-10
            lin_vel_z = -0.0
            feet_air_time = 1.5
            dof_pos_limits = -0.0
            feet_contact_forces = -0.02
            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.1
            dof_vel = -1e-5
            dof_acc = -5e-9
            orientation = 1.0
            base_height = 0.2

            termination = -0.0
            no_fly = 0.0
            ang_vel_xy = -0.0
            action_rate = -0.0
            stand_still = -0.0

            dof_ref_pos_diff = 2.2
            feet_distance = 0.2
            knee_distance = 0.2
            foot_slip = -0.1
            base_acc = 0.2
            vel_mismatch_exp = 0.5
            track_vel_hard = 0.5
            default_joint_pos = 0.8
            feet_height = -0.0
            low_speed = 0.2
            action_smoothness = -0.003
            feet_contact_number = 1.4
            feet_clearance = 1.6

    class sim(ConfigBipedRobot.sim):
        dt = 0.001  # 1000 Hz
