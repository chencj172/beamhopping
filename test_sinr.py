"""
快速测试脚本 - 检查SINR和data_rate的计算
"""

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.dynamic_satellite_env import DynamicSatelliteEnv

def test_sinr_and_data_rates():
    """测试SINR和数据率计算"""
    print("创建环境...")
    env = DynamicSatelliteEnv(
        num_beams=4,
        num_cells=19,
        enable_channel_dynamics=True,
        enable_traffic_dynamics=True,
        enable_weather=False,
        enable_mobility=False,
        traffic_intensity=0.7
    )
    
    print("重置环境...")
    obs = env.reset()
    
    # 创建简单的动作
    action = {
        'beam_allocation': [0, 1, 2, 3],  # 前4个小区
        'power_allocation': [0.25, 0.25, 0.25, 0.25]  # 均匀功率分配
    }
    
    print("执行第一步...")
    obs, reward, done, info = env.step(action)
    print(f"第1步 - 奖励: {reward:.3f}, 服务的数据包: {np.sum(info['served_per_cell'])}")
    
    print("\n执行第二步...")
    obs, reward, done, info = env.step(action)
    print(f"第2步 - 奖励: {reward:.3f}, 服务的数据包: {np.sum(info['served_per_cell'])}")
    
    print("\n执行第三步...")
    obs, reward, done, info = env.step(action)
    print(f"第3步 - 奖励: {reward:.3f}, 服务的数据包: {np.sum(info['served_per_cell'])}")
    
if __name__ == "__main__":
    test_sinr_and_data_rates()
