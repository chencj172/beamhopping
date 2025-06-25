"""
快速优化训练脚本 - 针对性解决PPO性能问题

关键改进：
1. 增大学习率和探索
2. 改进奖励塑形
3. 增加训练episodes
4. 优化网络更新频率
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from environment.satellite_env import SatelliteEnv
from rl.ppo import PPO
from utils.visualization import Visualization
from utils.metrics import Metrics


def main():
    """快速优化训练"""
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=== PPO性能优化训练 ===")
    
    # 优化参数设置
    num_episodes = 3000  # 增加训练episodes
    max_steps = 100
    
    # 创建环境
    env = SatelliteEnv(
        num_beams=4,
        num_cells=19,
        total_power_dbw=39.0,
        total_bandwidth_mhz=200.0,
        satellite_height_km=550.0,
        max_queue_size=200,
        packet_ttl=15,
        reward_throughput_weight=0.8,  # 增加吞吐量权重
        reward_delay_weight=0.2
    )
    
    env.max_steps = max_steps
    
    # 设置优化的PPO代理
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    agent = PPO(
        state_dim=state_dim,
        discrete_action_dim=4,
        continuous_action_dim=4,
        discrete_action_space_size=19,
        lr_actor=2e-3,        # 增大学习率
        lr_critic=2e-3,       # 增大学习率
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.08,    # 增大探索
        update_epochs=20,     # 增加更新epochs
        mini_batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 创建输出目录
    os.makedirs('optimized_models', exist_ok=True)
    os.makedirs('optimized_results', exist_ok=True)
    
    # 训练指标
    rewards = []
    throughputs = []
    delays = []
    packet_losses = []
    fairness_scores = []
    
    best_reward = -float('inf')
    best_avg_reward = -float('inf')
    
    # 训练循环
    for episode in tqdm(range(num_episodes), desc='优化训练'):
        # 重置环境
        state = env.reset()
        state = state.flatten()
        
        episode_reward = 0
        episode_throughput = []
        episode_delays = []
        episode_packet_losses = []
        episode_fairness = []
        
        # 运行episode
        for step in range(max_steps):
            # 选择动作
            action, log_prob, value, entropy = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 奖励塑形 - 增强学习信号
            shaped_reward = reward
            
            # 额外奖励塑形
            if 'throughput' in info and info['throughput'] > 0:
                # 奖励高吞吐量
                shaped_reward += 0.1 * min(info['throughput'] / 10.0, 1.0)
                
            if 'data_rates' in info and len(info['data_rates']) > 0:
                # 奖励公平性
                fairness = Metrics.calculate_fairness_index(info['data_rates'])
                shaped_reward += 0.05 * fairness
                episode_fairness.append(fairness)
            
            if 'delay' in info and info['delay'] > 0:
                # 惩罚高延迟
                shaped_reward -= 0.05 * min(info['delay'] / 10.0, 1.0)
                episode_delays.append(info['delay'])
                
            if 'packet_loss' in info:
                # 严重惩罚丢包
                shaped_reward -= 0.2 * info['packet_loss']
                episode_packet_losses.append(info['packet_loss'])
              # 存储经验
            next_state_flat = next_state.flatten()
            agent.buffer.add(state, action, shaped_reward, log_prob, value, done)
            
            # 记录指标
            episode_reward += shaped_reward
            if 'throughput' in info:
                episode_throughput.append(info['throughput'])
            
            state = next_state_flat
            
            if done:
                break
        
        # 更新网络 - 更频繁的更新
        if len(agent.buffer) >= agent.mini_batch_size:
            agent.update()
        
        # 记录episode指标
        rewards.append(episode_reward)
        throughputs.append(np.mean(episode_throughput) if episode_throughput else 0)
        delays.append(np.mean(episode_delays) if episode_delays else 0)
        packet_losses.append(np.mean(episode_packet_losses) if episode_packet_losses else 0)
        fairness_scores.append(np.mean(episode_fairness) if episode_fairness else 0)
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('optimized_models/ppo_model_best.pt')
        
        # 检查性能提升
        if episode >= 100:
            recent_avg = np.mean(rewards[-100:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                agent.save('optimized_models/ppo_model_best_avg.pt')
        
        # 定期保存和评估
        if (episode + 1) % 500 == 0:
            agent.save(f'optimized_models/ppo_model_episode_{episode + 1}.pt')
            
            # 打印进度
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            avg_throughput = np.mean(throughputs[-100:]) if len(throughputs) >= 100 else np.mean(throughputs)
            avg_delay = np.mean(delays[-100:]) if len(delays) >= 100 else np.mean(delays)
            avg_fairness = np.mean(fairness_scores[-100:]) if len(fairness_scores) >= 100 else np.mean(fairness_scores)
            
            print(f"\nEpisode {episode + 1}/{num_episodes}:")
            print(f"  近100 episodes平均奖励: {avg_reward:.4f}")
            print(f"  近100 episodes平均吞吐量: {avg_throughput:.4f}")
            print(f"  近100 episodes平均延迟: {avg_delay:.4f}")
            print(f"  近100 episodes平均公平性: {avg_fairness:.4f}")
            print(f"  历史最佳奖励: {best_reward:.4f}")
    
    # 保存最终模型
    agent.save('optimized_models/ppo_model_final_optimized.pt')
    
    # 绘制训练结果
    plt.figure(figsize=(20, 12))
    
    # 奖励曲线
    plt.subplot(2, 4, 1)
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(rewards)), moving_avg, label='Moving Avg (100)', linewidth=2)
    plt.title('优化训练奖励曲线')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 吞吐量曲线
    plt.subplot(2, 4, 2)
    plt.plot(throughputs)
    if len(throughputs) >= 100:
        moving_avg = np.convolve(throughputs, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(throughputs)), moving_avg, linewidth=2)
    plt.title('吞吐量变化')
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    plt.grid(True, alpha=0.3)
    
    # 延迟曲线
    plt.subplot(2, 4, 3)
    plt.plot(delays)
    if len(delays) >= 100:
        moving_avg = np.convolve(delays, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(delays)), moving_avg, linewidth=2)
    plt.title('延迟变化')
    plt.xlabel('Episode')
    plt.ylabel('Delay')
    plt.grid(True, alpha=0.3)
    
    # 丢包率曲线
    plt.subplot(2, 4, 4)
    plt.plot(packet_losses)
    if len(packet_losses) >= 100:
        moving_avg = np.convolve(packet_losses, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(packet_losses)), moving_avg, linewidth=2)
    plt.title('丢包率变化')
    plt.xlabel('Episode')
    plt.ylabel('Packet Loss')
    plt.grid(True, alpha=0.3)
    
    # 公平性曲线
    plt.subplot(2, 4, 5)
    plt.plot(fairness_scores)
    if len(fairness_scores) >= 100:
        moving_avg = np.convolve(fairness_scores, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(fairness_scores)), moving_avg, linewidth=2)
    plt.title('公平性变化')
    plt.xlabel('Episode')
    plt.ylabel('Fairness Index')
    plt.grid(True, alpha=0.3)
    
    # 最近1000 episodes的分布
    plt.subplot(2, 4, 6)
    recent_rewards = rewards[-1000:] if len(rewards) >= 1000 else rewards
    plt.hist(recent_rewards, bins=50, alpha=0.7)
    plt.title('最近奖励分布')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 性能对比（与之前的训练对比）
    plt.subplot(2, 4, 7)
    # 显示关键指标的最终性能
    final_metrics = ['Avg Reward', 'Avg Throughput', 'Avg Delay', 'Avg Fairness']
    final_values = [
        np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        np.mean(throughputs[-100:]) if len(throughputs) >= 100 else np.mean(throughputs),
        np.mean(delays[-100:]) if len(delays) >= 100 else np.mean(delays),
        np.mean(fairness_scores[-100:]) if len(fairness_scores) >= 100 else np.mean(fairness_scores)
    ]
    plt.bar(final_metrics, final_values)
    plt.title('最终性能指标')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 学习进度
    plt.subplot(2, 4, 8)
    # 按阶段显示平均奖励
    stage_size = max(1, num_episodes // 10)
    stage_rewards = []
    for i in range(0, num_episodes, stage_size):
        stage_avg = np.mean(rewards[i:i+stage_size])
        stage_rewards.append(stage_avg)
    plt.plot(range(len(stage_rewards)), stage_rewards, 'o-', linewidth=2, markersize=8)
    plt.title('学习进度（分阶段平均）')
    plt.xlabel('Training Stage')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_results/optimization_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出最终统计
    print("\n" + "="*80)
    print("优化训练完成！")
    print("="*80)
    print(f"总训练episodes: {num_episodes}")
    print(f"最终平均奖励 (最近100 episodes): {np.mean(rewards[-100:]):.4f}")
    print(f"最终平均吞吐量 (最近100 episodes): {np.mean(throughputs[-100:]):.4f}")
    print(f"最终平均延迟 (最近100 episodes): {np.mean(delays[-100:]):.4f}")
    print(f"最终平均公平性 (最近100 episodes): {np.mean(fairness_scores[-100:]):.4f}")
    print(f"历史最佳单episode奖励: {best_reward:.4f}")
    print(f"历史最佳100 episodes平均奖励: {best_avg_reward:.4f}")
    print("\n模型保存位置:")
    print("  - 最佳单episode模型: optimized_models/ppo_model_best.pt")
    print("  - 最佳平均性能模型: optimized_models/ppo_model_best_avg.pt") 
    print("  - 最终模型: optimized_models/ppo_model_final_optimized.pt")
    print("="*80)


if __name__ == "__main__":
    main()
