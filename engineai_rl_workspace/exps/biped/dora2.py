from engineai_gym.envs.robots.biped.dora2.config_dora2_shoes_rough import ConfigDora2ShoesRough
from engineai_rl_workspace.utils.exp_registry import exp_registry

from engineai_gym.envs.robots.biped.dora2.config_dora2_rough import ConfigDora2Rough
from engineai_rl.algos.ppo import Ppo, PpoContact
from engineai_rl.exps.biped.dora2.config_dora2_ppo import ConfigDora2Ppo
from engineai_rl.exps.biped.dora2.config_dora2_ppo_contact import ConfigDora2PpoContact
from engineai_gym.envs.robots.biped.dora2.dora2 import Dora2
from engineai_gym.envs.robots.biped.rewards_biped import RewardsBiped
from engineai_gym.envs.robots.biped.goals_biped import GoalsBiped
from engineai_gym.envs.robots.biped.dora2.obs_dora2 import ObsDora2

exp_registry.register(
    name="dora2_rough_ppo",
    task_class=Dora2,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigDora2Rough(),
    algo_class=Ppo,
    algo_cfg=ConfigDora2Ppo(),
)

exp_registry.register(
    name="dora2_rough_ppo_contact",
    task_class=Dora2,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigDora2Rough(),
    algo_class=PpoContact,
    algo_cfg=ConfigDora2PpoContact(),
    obs_class=ObsDora2,
)

exp_registry.register(
    name="dora2_shoes_rough_ppo_contact",
    task_class=Dora2,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigDora2ShoesRough(),
    algo_class=PpoContact,
    algo_cfg=ConfigDora2PpoContact(),
    obs_class=ObsDora2,
)

exp_registry.register(
    name="dora2_shoes_rough_ppo",
    task_class=Dora2,
    goal_class=GoalsBiped,
    reward_class=RewardsBiped,
    env_cfg=ConfigDora2ShoesRough(),
    algo_class=Ppo,
    algo_cfg=ConfigDora2Ppo(),
)