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