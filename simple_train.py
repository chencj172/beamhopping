"""
简化的PPO训练脚本 - 修复API问题
使用新的改进奖励函数
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment.satellite_env import SatelliteEnv
from rl.ppo import PPO
from utils.metrics import Metrics


def simple_train():
    """简化的训练函数"""
    
    print("=== 简化PPO训练 (新奖励函数) ===")
    
    # 设置参数
    num_episodes = 1000
    max_steps = 100
    
    # 创建环境
    env = SatelliteEnv()
    env.max_steps = max_steps
    
    # 创建PPO代理
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    agent = PPO(
        state_dim=state_dim,
        discrete_action_dim=4,
        continuous_action_dim=4,
        discrete_action_space_size=19,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.05,
        update_epochs=10,
        mini_batch_size=64
    )
    
    # 创建输出目录
    os.makedirs('simple_results', exist_ok=True)
    
    # 训练指标
    rewards = []
    throughputs = []
    delays = []
    
    # 训练循环
    for episode in tqdm(range(num_episodes), desc='简化训练'):
        state = env.reset()
        state = state.flatten()
        
        episode_reward = 0
        episode_throughput = []
        episode_delays = []
        
        for step in range(max_steps):
            # 选择动作
            action, discrete_log_prob, continuous_log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验 - 使用正确的buffer.add调用
            agent.buffer.add(
                state,
                action['beam_allocation'],
                action['power_allocation'], 
                discrete_log_prob,
                continuous_log_prob,
                reward,
                value,
                done
            )
            
            # 记录指标
            episode_reward += reward
            if 'throughput' in info:
                episode_throughput.append(info['throughput'])
            if 'delay' in info:
                episode_delays.append(info['delay'])
            
            # 更新状态
            state = next_state.flatten()
            
            if done:
                break
        
        # 更新网络
        if len(agent.buffer) >= agent.mini_batch_size:
            agent.update()
        
        # 记录episode指标
        rewards.append(episode_reward)
        throughputs.append(np.mean(episode_throughput) if episode_throughput else 0)
        delays.append(np.mean(episode_delays) if episode_delays else 0)
        
        # 定期输出进度
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_throughput = np.mean(throughputs[-100:])
            avg_delay = np.mean(delays[-100:])
            
            print(f"\nEpisode {episode + 1}/{num_episodes}:")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均吞吐量: {avg_throughput:.4f}")
            print(f"  平均延迟: {avg_delay:.4f}")
    
    # 保存模型
    agent.save('simple_results/ppo_model_simple.pt')
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.3)
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), moving_avg, linewidth=2)
    plt.title('奖励曲线 (新奖励函数)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(throughputs)
    plt.title('吞吐量')
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    
    plt.subplot(1, 3, 3)
    plt.plot(delays)
    plt.title('延迟')
    plt.xlabel('Episode')
    plt.ylabel('Delay')
    
    plt.tight_layout()
    plt.savefig('simple_results/simple_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出最终结果
    print("\n" + "="*60)
    print("简化训练完成！")
    print(f"最终平均奖励: {np.mean(rewards[-100:]):.4f}")
    print(f"最终平均吞吐量: {np.mean(throughputs[-100:]):.4f}")
    print(f"最终平均延迟: {np.mean(delays[-100:]):.4f}")
    print("="*60)


if __name__ == "__main__":
    simple_train()
