import mujoco
import numpy as np
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/mmlab-rl/codes/engineai_rl_workspace/engineai_gym/engineai_gym/resources/robots/biped/dora2/mjcf/dora2_shoes.xml", help="Path to the MuJoCo model XML file")
    parser.add_argument("--show_values", action="store_true", help="Show qpos and qvel values")
    parser.add_argument("--simulate", action="store_true", help="Run simulation and show joint values")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    return parser.parse_args()

def print_joint_info(model, data=None):
    # 打印关节信息
    print("\n关节信息:")
    print(f"关节总数: {model.njnt}")
    
    # 收集关节信息
    joint_info_list = []
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        joint_type_str = ["free", "ball", "slide", "hinge"][joint_type]
        joint_qpos_addr = model.jnt_qposadr[i]
        joint_axis = model.jnt_axis[i]
        
        info = f"ID: {i}, 名称: {joint_name}, 类型: {joint_type_str}, qpos地址: {joint_qpos_addr}, 轴: {joint_axis}"
        
        # 如果提供了data，显示qpos和qvel的值
        if data is not None:
            if joint_type == 0:  # free joint
                qpos_vals = data.qpos[joint_qpos_addr:joint_qpos_addr+7]
                info += f", qpos值: {qpos_vals}"
            elif joint_type == 1:  # ball joint
                qpos_vals = data.qpos[joint_qpos_addr:joint_qpos_addr+4]
                info += f", qpos值: {qpos_vals}"
            elif joint_type in [2, 3]:  # slide or hinge joint
                qpos_val = data.qpos[joint_qpos_addr]
                qvel_idx = model.jnt_dofadr[i]
                qvel_val = data.qvel[qvel_idx] if qvel_idx < len(data.qvel) else None
                info += f", qpos值: {qpos_val}, qvel值: {qvel_val}"
        
        joint_info_list.append((i, joint_name, info))
    
    # 打印关节名称和ID
    print("\n关节名称和ID:")
    for _, _, info in joint_info_list:
        print(info)
    
    # 打印关节顺序列表
    print("\n关节顺序列表 (按ID排序):")
    joint_names = [name for _, name, _ in joint_info_list]
    print(", ".join(joint_names))
    
    # 打印控制关节列表（排除free joint）
    control_joint_names = [name for i, name, _ in joint_info_list if model.jnt_type[i] != 0]
    print("\n控制关节列表 (排除free joint):")
    print(", ".join(control_joint_names))
    
    # 打印Python列表格式的关节名称
    print("\nPython列表格式的关节名称:")
    print(f"all_joint_names = {joint_names}")
    print(f"control_joint_names = {control_joint_names}")
    
    return joint_names, control_joint_names

def print_actuator_info(model):
    # 打印执行器信息
    print("\n执行器信息:")
    print(f"执行器总数: {model.nu}")
    
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        actuator_trnid = model.actuator_trnid[i]
        actuator_joint = model.joint(actuator_trnid[0]).name if actuator_trnid[0] >= 0 else "None"
        
        # 获取执行器的控制范围
        ctrl_range = model.actuator_ctrlrange[i]
        ctrl_range_str = f"[{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}]" if ctrl_range[0] != ctrl_range[1] else "无限制"
        
        # 获取执行器的力/扭矩范围
        force_range = model.actuator_forcerange[i]
        force_range_str = f"[{force_range[0]:.2f}, {force_range[1]:.2f}]" if force_range[0] != force_range[1] else "无限制"
        
        # 获取执行器的增益和偏置
        gain = model.actuator_gainprm[i][0]
        bias = model.actuator_biasprm[i][1]
        
        print(f"ID: {i}, 名称: {actuator_name}, 控制关节: {actuator_joint}")
        print(f"  控制范围: {ctrl_range_str}, 力/扭矩范围: {force_range_str}")
        print(f"  增益: {gain:.4f}, 偏置: {bias:.4f}")

def print_joint_limits(model):
    # 打印关节限制信息
    print("\n关节限制信息:")
    
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        
        # 只有铰链和滑动关节有限制
        if joint_type in [2, 3]:  # slide or hinge joint
            qpos_addr = model.jnt_qposadr[i]
            joint_range = model.jnt_range[i]
            
            # 检查是否有限制
            if joint_range[0] != joint_range[1]:
                print(f"{joint_name}: 范围 [{joint_range[0]:.2f}, {joint_range[1]:.2f}] rad")
            else:
                print(f"{joint_name}: 无限制")
        elif joint_type == 0:  # free joint
            print(f"{joint_name}: 自由关节，无限制")
        elif joint_type == 1:  # ball joint
            print(f"{joint_name}: 球关节，无限制")

def print_state_space_info(model):
    # 打印qpos和qvel的索引信息
    print("\nqpos索引信息:")
    print(f"qpos维度: {model.nq}")
    
    qpos_indices = {}
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        qpos_addr = model.jnt_qposadr[i]
        
        if joint_type == 0:  # free joint
            qpos_indices[joint_name] = list(range(qpos_addr, qpos_addr+7))
            print(f"{joint_name}: qpos[{qpos_addr}:{qpos_addr+7}]")
        elif joint_type == 1:  # ball joint
            qpos_indices[joint_name] = list(range(qpos_addr, qpos_addr+4))
            print(f"{joint_name}: qpos[{qpos_addr}:{qpos_addr+4}]")
        elif joint_type in [2, 3]:  # slide or hinge joint
            qpos_indices[joint_name] = [qpos_addr]
            print(f"{joint_name}: qpos[{qpos_addr}]")
    
    print("\nqvel索引信息:")
    print(f"qvel维度: {model.nv}")
    
    qvel_indices = {}
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        dof_addr = model.jnt_dofadr[i]
        
        if joint_type == 0:  # free joint
            qvel_indices[joint_name] = list(range(dof_addr, dof_addr+6))
            print(f"{joint_name}: qvel[{dof_addr}:{dof_addr+6}]")
        elif joint_type == 1:  # ball joint
            qvel_indices[joint_name] = list(range(dof_addr, dof_addr+3))
            print(f"{joint_name}: qvel[{dof_addr}:{dof_addr+3}]")
        elif joint_type in [2, 3]:  # slide or hinge joint
            qvel_indices[joint_name] = [dof_addr]
            print(f"{joint_name}: qvel[{dof_addr}]")
    
    return qpos_indices, qvel_indices

def main():
    args = get_args()
    
    # 加载MuJoCo模型
    print(f"加载模型: {args.model_path}")
    model = mujoco.MjModel.from_xml_path(args.model_path)
    data = mujoco.MjData(model)
    
    # 初始化模型
    mujoco.mj_forward(model, data)
    
    # 打印自由度信息
    print("\n自由度(DOF)信息:")
    print(f"自由度总数: {model.nv}")
    print(f"位置状态维度: {model.nq}")
    
    # 打印关节信息
    if args.show_values:
        print_joint_info(model, data)
    else:
        print_joint_info(model)
    
    # 打印状态空间信息
    qpos_indices, qvel_indices = print_state_space_info(model)
    
    # 打印关节限制信息
    print_joint_limits(model)
    
    # 打印执行器信息
    print_actuator_info(model)
    
    # 如果需要模拟，运行模拟并显示关节值
    if args.simulate:
        print("\n开始模拟...")
        
        # 创建可视化对象（如果需要）
        # 这里省略可视化代码，仅打印关节值
        
        for i in range(args.steps):
            # 应用一些随机控制信号
            data.ctrl[:] = np.random.uniform(-0.1, 0.1, size=model.nu)
            
            # 步进模拟
            mujoco.mj_step(model, data)
            
            if i % 10 == 0:  # 每10步打印一次
                print(f"\n步骤 {i}:")
                print_joint_info(model, data)
                time.sleep(0.5)  # 暂停以便查看输出

if __name__ == "__main__":
    main()