#!/usr/bin/env python3
"""
将tensorboard日志转换为wandb格式的脚本
"""

import os
import glob
import wandb
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from datetime import datetime

def extract_tensorboard_data(logdir):
    """从tensorboard日志文件中提取数据"""
    print(f"正在处理日志目录: {logdir}")
    
    # 查找所有事件文件
    event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        print(f"警告: 在 {logdir} 中未找到tensorboard事件文件")
        return None
    
    print(f"找到 {len(event_files)} 个事件文件")
    
    # 创建事件累积器
    ea = EventAccumulator(logdir)
    ea.Reload()
    
    # 获取所有标量标签
    scalar_tags = ea.Tags()['scalars']
    print(f"找到标量标签: {scalar_tags}")
    
    # 提取数据
    data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        timestamps = [event.wall_time for event in scalar_events]
        
        data[tag] = {
            'steps': steps,
            'values': values,
            'timestamps': timestamps
        }
    
    return data

def convert_to_wandb(logdir, project_name, run_name=None):
    """将tensorboard数据转换为wandb格式"""
    
    # 提取tensorboard数据
    tb_data = extract_tensorboard_data(logdir)
    if tb_data is None:
        return None
    
    # 如果没有指定运行名称，使用目录名
    if run_name is None:
        run_name = os.path.basename(logdir)
    
    # 设置离线模式
    os.environ['WANDB_MODE'] = 'offline'
    
    # 初始化wandb运行
    run = wandb.init(
        project=project_name,
        name=run_name,
        tags=['tensorboard_import', 'converted'],
        notes=f"从tensorboard日志转换: {logdir}"
    )
    
    print(f"创建wandb运行: {run_name}")
    
    # 获取所有步数的并集
    all_steps = set()
    for tag, data in tb_data.items():
        all_steps.update(data['steps'])
    
    all_steps = sorted(all_steps)
    
    # 按步数记录数据
    for step in all_steps:
        log_data = {'step': step}
        
        for tag, data in tb_data.items():
            # 找到最接近的步数
            if step in data['steps']:
                idx = data['steps'].index(step)
                log_data[tag] = data['values'][idx]
        
        # 只记录包含数据的步数
        if len(log_data) > 1:  # 除了step之外还有其他数据
            wandb.log(log_data, step=step)
    
    # 完成运行
    wandb.finish()
    
    return run

def main():
    parser = argparse.ArgumentParser(description='将tensorboard日志转换为wandb格式')
    parser.add_argument('--logs_dir', default='logs', help='tensorboard日志根目录')
    parser.add_argument('--project', default='engineai_dora2', help='wandb项目名称')
    parser.add_argument('--experiment', help='特定实验名称（可选）')
    
    args = parser.parse_args()
    
    print("=== Tensorboard to Wandb 转换工具 ===")
    print(f"日志目录: {args.logs_dir}")
    print(f"项目名称: {args.project}")
    
    # 查找所有实验目录
    if args.experiment:
        # 转换特定实验
        exp_dirs = [os.path.join(args.logs_dir, args.experiment)]
    else:
        # 转换所有实验
        exp_dirs = glob.glob(os.path.join(args.logs_dir, "*"))
        exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
    
    print(f"找到 {len(exp_dirs)} 个实验目录")
    
    converted_runs = []
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"\n处理实验: {exp_name}")
        
        # 查找该实验下的所有运行
        run_dirs = glob.glob(os.path.join(exp_dir, "*/20*"))  # 匹配时间戳目录
        if not run_dirs:
            run_dirs = glob.glob(os.path.join(exp_dir, "*"))
            run_dirs = [d for d in run_dirs if os.path.isdir(d)]
        
        for run_dir in run_dirs:
            run_name = f"{exp_name}_{os.path.basename(run_dir)}"
            print(f"  转换运行: {run_name}")
            
            try:
                run = convert_to_wandb(run_dir, args.project, run_name)
                if run:
                    converted_runs.append(run_name)
                    print(f"  ✓ 成功转换: {run_name}")
                else:
                    print(f"  ✗ 转换失败: {run_name}")
            except Exception as e:
                print(f"  ✗ 转换失败: {run_name} - {str(e)}")
    
    print(f"\n=== 转换完成 ===")
    print(f"成功转换 {len(converted_runs)} 个运行")
    
    # 显示如何同步到wandb.ai
    print("\n=== 如何同步到wandb.ai ===")
    print("1. 确保已登录wandb:")
    print("   wandb login")
    print("\n2. 同步所有离线运行:")
    print("   wandb sync wandb/offline-run-*")
    print("\n3. 或者同步特定运行:")
    wandb_dirs = glob.glob("wandb/offline-run-*")
    for wandb_dir in wandb_dirs[-3:]:  # 显示最后3个
        print(f"   wandb sync {wandb_dir}")

if __name__ == "__main__":
    main()
