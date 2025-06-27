"""
测试修复后的PPO NaN问题
"""
import numpy as np
import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.dynamic_satellite_env import DynamicSatelliteEnv
from rl.ppo import PPO

def test_nan_fixes():
    """测试NaN修复"""
    print("开始测试NaN修复...")
    
    # 创建环境
    env = DynamicSatelliteEnv(
        num_beams=3,
        num_cells=19,
        enable_channel_dynamics=True,
        enable_traffic_dynamics=True,
        enable_weather=True,
        enable_mobility=True,
        traffic_intensity=0.8
    )
    
    # 创建PPO智能体
    obs_dim = env.observation_space.shape[0]
    
    agent = PPO(
        state_dim=obs_dim,
        discrete_action_dim=env.num_beams,
        continuous_action_dim=env.num_beams,
        discrete_action_space_size=env.num_cells,
        lr_actor=2e-4,
        lr_critic=5e-4,
        device='cpu'  # 使用CPU避免GPU内存问题
    )
    
    # 重置环境
    state = env.reset()
    
    print("环境重置成功")
    print(f"观测空间维度: {state.shape}")
    
    # 测试几个步骤
    for step in range(5):
        print(f"\n=== 步骤 {step + 1} ===")
        
        # 选择动作
        try:
            action, discrete_log_prob, continuous_log_prob, state_value = agent.select_action(state)
            print("动作选择成功")
            print(f"波束分配: {action['beam_allocation']}")
            print(f"功率分配: {action['power_allocation']}")
        except Exception as e:
            print(f"动作选择失败: {e}")
            break
        
        # 执行动作
        try:
            next_state, reward, done, info = env.step(action)
            print(f"环境步进成功，奖励: {reward:.4f}")
            
            # 检查状态是否包含NaN
            if np.any(np.isnan(next_state)):
                print("警告: 状态包含NaN值")
            else:
                print("状态正常")
                
            state = next_state
            
        except Exception as e:
            print(f"环境步进失败: {e}")
            break
        
        if done:
            break
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_nan_fixes()
