# Train
```bash
python engineai_rl_workspace/scripts/train.py --exp_name dora2_rough_ppo --headless --logger tensorboard
```

# Rusume training
```bash
python engineai_rl_workspace/scripts/train.py --exp_name dora2_rough_ppo --headless --logger tensorboard --resume --load_run <run_name>
```

# Play with recorded video
```bash
python engineai_rl_workspace/scripts/play.py --exp_name dora2_rough_ppo --video --num_envs 1 --headless --record_length 1000 --test_length 1500 --load_run <run_name>
```

```bash
python engineai_rl_workspace/scripts/play.py --exp_name dora2_shoes_rough_ppo_contact --video --num_envs 1 --headless --record_length 1500 --test_length 1500 --load_run <run_name>
```

# Export policy
```bash
python engineai_rl_workspace/scripts/export_policy.py --exp_name dora2_shoes_rough_ppo_contact --load_run <run_name>
```

# Simulation in Mujoco
```bash
python engineai_rl_workspace/scripts/sim_mujoco.py --exp_name dora2_shoes_rough_ppo_contact --load_run <run_name>
```

# Update git info
```bash
python engineai_rl_workspace/scripts/update_git_info.py --exp_name dora2_shoes_rough_ppo_contact --load_run <run_name> --commit <commit_hash>
```