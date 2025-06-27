"""
PPO延迟优化训练脚本
专门解决PPO延迟过高的问题
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.dynamic_satellite_env import DynamicSatelliteEnv
from rl.ppo import PPO

def train_delay_optimized_ppo():
    """训练延迟优化的PPO"""
    print("开始训练延迟优化的PPO...")
    
    # 创建环境，专门优化延迟
    env = DynamicSatelliteEnv(
        num_beams=3,
        num_cells=19,
        enable_channel_dynamics=True,
        enable_traffic_dynamics=True,
        enable_weather=True,
        enable_mobility=True,
        traffic_intensity=0.8,
        reward_weights={
            'throughput': 0.35,
            'delay': 0.40,        # 延迟权重最高
            'fairness': 0.15,
            'energy_efficiency': 0.10
        }
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
        gamma=0.995,
        clip_ratio=0.15,
        entropy_coef=0.005,
        update_epochs=6
    )
    
    # 训练参数
    episodes = 1200
    episode_rewards = []
    episode_delays = []
    episode_throughputs = []
    
    print("开始训练...")
    for episode in tqdm(range(episodes)):
        obs = env.reset()
        episode_reward = 0
        episode_info = []
        
        for step in range(200):
            action, discrete_log_prob, continuous_log_prob, state_value = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.buffer.add(
                obs, action['beam_allocation'], action['power_allocation'],
                discrete_log_prob, continuous_log_prob, reward, state_value, done
            )
            
            episode_reward += reward
            episode_info.append(info)
            obs = next_obs
            
            if done:
                break
        
        # 更新智能体
        if len(agent.buffer) >= agent.mini_batch_size:
            agent.update()
        
        # 记录统计
        episode_rewards.append(episode_reward)
        avg_delay = np.mean([info['delay'] for info in episode_info])
        avg_throughput = np.mean([info['throughput'] for info in episode_info])
        episode_delays.append(avg_delay)
        episode_throughputs.append(avg_throughput)
        
        # 每100轮打印一次
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Reward={np.mean(episode_rewards[-100:]):.3f}, "
                  f"Delay={np.mean(episode_delays[-100:]):.3f}, "
                  f"Throughput={np.mean(episode_throughputs[-100:]):.3f}")
    
    # 保存模型
    os.makedirs('delay_optimized_models', exist_ok=True)
    agent.save('delay_optimized_models/ppo_delay_optimized.pt')
    
    # 绘制训练结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_delays)
    plt.title('Average Delays')
    plt.xlabel('Episode')
    plt.ylabel('Delay')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(episode_throughputs)
    plt.title('Average Throughputs')
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('delay_optimized_results', exist_ok=True)
    plt.savefig('delay_optimized_results/training_curves.png', dpi=300)
    plt.show()
    
    print("训练完成！模型已保存到 delay_optimized_models/ppo_delay_optimized.pt")
    return agent

if __name__ == "__main__":
    train_delay_optimized_ppo()
