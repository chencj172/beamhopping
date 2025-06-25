#!/usr/bin/env python3
"""
测试 traffic_intensity 参数应用
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from environment.satellite_env import SatelliteEnv
import time


def test_traffic_intensity():
    """测试流量强度参数的应用"""
    print("=== 测试 Traffic Intensity 参数应用 ===")
    
    # 测试不同的流量强度
    traffic_intensities = [0.2, 0.5, 0.8]
    
    for intensity in traffic_intensities:
        print(f"\n--- 测试流量强度: {intensity} ---")
        
        # 创建环境
        env = SatelliteEnv(
            num_beams=4,
            num_cells=19,
            traffic_intensity=intensity
        )
        
        # 重置环境
        state = env.reset()
        
        # 运行几个步骤来生成流量并观察到达率
        arrivals_history = []
        
        for step in range(10):
            # 创建简单的动作
            action = {
                'beam_allocation': np.random.randint(0, 19, size=4),
                'power_allocation': np.ones(4) / 4
            }
            
            # 执行步骤
            next_state, reward, done, info = env.step(action)
            
            # 生成流量并记录到达数量
            arrivals = env.queue_model.generate_traffic()
            arrivals_history.append(arrivals)
            
            if done:
                break
        
        # 计算平均到达率
        total_arrivals = np.sum(arrivals_history, axis=0)
        avg_arrivals_per_cell = np.mean(total_arrivals)
        
        print(f"流量强度: {intensity}")
        print(f"平均每小区到达数据包数: {avg_arrivals_per_cell:.3f}")
        print(f"队列模型的流量强度: {env.queue_model.traffic_intensity}")
        print(f"队列模型的到达率范围: [{np.min(env.queue_model.arrival_rates):.3f}, {np.max(env.queue_model.arrival_rates):.3f}]")


def test_dynamic_traffic_intensity():
    """测试动态调整流量强度"""
    print("\n\n=== 测试动态调整流量强度 ===")
    
    # 创建环境
    env = SatelliteEnv(
        num_beams=4,
        num_cells=19,
        traffic_intensity=0.3
    )
    
    # 测试动态调整
    intensities = [0.3, 0.6, 0.9]
    
    for intensity in intensities:
        print(f"\n--- 设置流量强度为: {intensity} ---")
        
        # 动态调整流量强度
        env.set_traffic_intensity(intensity)
        
        # 重置环境以应用新参数
        env.reset()
        
        # 运行几步观察效果
        arrivals_total = np.zeros(19)
        
        for step in range(5):
            action = {
                'beam_allocation': np.random.randint(0, 19, size=4),
                'power_allocation': np.ones(4) / 4
            }
            
            next_state, reward, done, info = env.step(action)
            arrivals = env.queue_model.generate_traffic()
            arrivals_total += arrivals
        
        avg_arrivals = np.mean(arrivals_total)
        print(f"设置后的流量强度: {env.traffic_intensity}")
        print(f"队列模型的流量强度: {env.queue_model.traffic_intensity}")
        print(f"平均到达数据包数: {avg_arrivals:.3f}")


def test_curriculum_learning_simulation():
    """模拟课程学习中的流量强度变化"""
    print("\n\n=== 模拟课程学习流量强度变化 ===")
    
    # 创建环境
    env = SatelliteEnv(
        num_beams=4,
        num_cells=19,
        traffic_intensity=0.3
    )
    
    # 模拟课程学习的不同阶段
    stages = [
        {'stage': 1, 'traffic_intensity': 0.3},
        {'stage': 2, 'traffic_intensity': 0.5},
        {'stage': 3, 'traffic_intensity': 0.7},
        {'stage': 4, 'traffic_intensity': 0.9}
    ]
    
    for stage_info in stages:
        stage = stage_info['stage']
        intensity = stage_info['traffic_intensity']
        
        print(f"\n--- 课程学习阶段 {stage}: 流量强度 {intensity} ---")
        
        # 设置新的流量强度
        env.set_traffic_intensity(intensity)
        env.reset()
        
        # 记录几个步骤的统计信息
        rewards = []
        throughputs = []
        delays = []
        
        for episode in range(3):  # 模拟几个episode
            state = env.reset()
            episode_reward = 0
            episode_throughput = 0
            episode_delay = 0
            
            for step in range(20):  # 每个episode 20步
                action = {
                    'beam_allocation': np.random.randint(0, 19, size=4),
                    'power_allocation': np.ones(4) / 4
                }
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                if 'throughput' in info:
                    episode_throughput += info['throughput']
                if 'delay' in info:
                    episode_delay += info['delay']
                
                if done:
                    break
            
            rewards.append(episode_reward)
            throughputs.append(episode_throughput / step if step > 0 else 0)
            delays.append(episode_delay / step if step > 0 else 0)
        
        print(f"平均奖励: {np.mean(rewards):.3f}")
        print(f"平均吞吐量: {np.mean(throughputs):.3f}")
        print(f"平均延迟: {np.mean(delays):.3f}")


if __name__ == '__main__':
    test_traffic_intensity()
    test_dynamic_traffic_intensity()
    test_curriculum_learning_simulation()
    print("\n=== 测试完成 ===")
