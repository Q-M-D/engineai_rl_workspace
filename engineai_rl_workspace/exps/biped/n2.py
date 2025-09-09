from engineai_rl_workspace.utils.exp_registry import exp_registry

from engineai_gym.envs.robots.biped.n2.config_n2_rough import ConfigN2Rough
from engineai_rl.algos.ppo import Ppo
from engineai_rl.exps.biped.n2.config_n2_ppo import ConfigN2Ppo
from engineai_gym.envs.robots.biped.n2.n2 import N2
from engineai_gym.envs.robots.biped.n2.reward_n2 import RewardN2
from engineai_gym.envs.robots.biped.goals_biped import GoalsBiped
from engineai_gym.envs.robots.biped.n2.obs_n2 import ObsN2

exp_registry.register(
    name="n2_rough_ppo",
    task_class=N2,
    goal_class=GoalsBiped,
    reward_class=RewardN2,
    env_cfg=ConfigN2Rough(),
    algo_class=Ppo,
    algo_cfg=ConfigN2Ppo(),
    obs_class=ObsN2,
)