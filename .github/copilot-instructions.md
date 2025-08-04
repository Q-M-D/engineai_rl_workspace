# EngineAI RL Workspace - AI Coding Agent Instructions

## Architecture Overview

This is a modular reinforcement learning framework for legged robots with three main packages:
- **`engineai_gym/`**: Environment definitions, robots, rewards, observations, domain randomization
- **`engineai_rl/`**: RL algorithms (PPO variants), neural networks, runners, storage systems  
- **`engineai_rl_workspace/`**: Experiment registry, training/evaluation scripts, utilities

The system uses a **registry pattern** for experiments - all experiments are registered in `engineai_rl_workspace/exps/` using `exp_registry.register()` with task classes, reward classes, environment configs, and algorithm configs.

## Key Workflows

### Training
```bash
python engineai_rl_workspace/scripts/train.py --exp_name dora2_rough_ppo --headless --logger tensorboard
```

### Resume Training
```bash  
python engineai_rl_workspace/scripts/train.py --exp_name dora2_rough_ppo --resume --load_run <run_name>
```

### Evaluation/Play
```bash
python engineai_rl_workspace/scripts/play.py --exp_name dora2_rough_ppo --video --num_envs 1 --load_run <run_name>
```

### Export Models
```bash
python engineai_rl_workspace/scripts/export_policy.py --exp_name dora2_rough_ppo --load_run <run_name>
```

## Critical Patterns

### Configuration System
- All configs inherit from `BaseConfig` which auto-instantiates nested classes
- Environment configs in `engineai_gym/envs/robots/*/config_*.py`
- Algorithm configs in `engineai_rl/exps/*/config_*.py`
- Configs are hierarchical with nested classes for different components

### Experiment Registration
```python
exp_registry.register(
    name="dora2_rough_ppo",
    task_class=Dora2,           # Robot environment
    goal_class=GoalsBiped,      # Goal generation
    reward_class=RewardsBiped,  # Reward computation  
    env_cfg=ConfigDora2Rough(), # Environment config
    algo_class=Ppo,             # RL algorithm
    algo_cfg=ConfigDora2Ppo()   # Algorithm config
)
```

### Multi-GPU Support
- Uses PyTorch distributed training with `WORLD_SIZE` environment variable
- PPO algorithm has built-in gradient reduction across GPUs
- Training script handles distributed initialization and barriers

### Git-Based Run Management
- **Every training run automatically saves git state** (commit, patches, config files)
- Resume functionality checks out exact code state from training
- Uses Redis locks to prevent concurrent modifications during resume
- All run artifacts stored in timestamped directories under `logs/`

## Development Conventions

### File Organization
- Robot-specific code: `engineai_gym/envs/robots/{biped,quadruped}/`
- Algorithm implementations: `engineai_rl/algos/`  
- Experiment definitions: `engineai_rl_workspace/exps/{biped,quadruped}/`
- Shared utilities: `engineai_rl_lib/`

### Modular Design Philosophy
- **Environments are composition of classes**: Task + Obs + Goals + DomainRands + Rewards
- **Algorithms are pluggable**: Same environment can use different RL algorithms
- **Networks are configurable**: Actor/critic networks defined separately from algorithms
- **Inheritance over configuration**: Extend base classes rather than complex config files

### Key Integration Points
- `VecGymWrapper` connects gym environments to RL algorithms
- `RolloutStorage` handles experience collection and batching
- `OnPolicyRunner` orchestrates training loop between environment and algorithm
- `InputRetrivalEnvWrapper` standardizes observation/action interfaces

When modifying algorithms, focus on the `update()` method in PPO classes. When adding robots, create new experiment registrations following the biped/quadruped patterns. All new experiments must be registered in the appropriate `exps/` subdirectory to be discoverable.
