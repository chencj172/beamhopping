"""
改进的训练脚本 - 优化PPO性能

主要改进：
1. 调整超参数以提高学习效率
2. 改进奖励函数设计
3. 增加课程学习
4. 添加更好的探索策略
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
from data.satellite_data_fixed import SatelliteDataFixed
from data.cell_data_fixed import CellDataFixed
from data.traffic_data_custom import TrafficDataCustom
from data.traffic_data_poisson import TrafficDataPoisson
from utils.visualization import Visualization
from utils.metrics import Metrics


def parse_args():
    """
    解析命令行参数 - 优化后的参数设置
    
    Returns:
        args: 参数对象
    """
    parser = argparse.ArgumentParser(description='Improved Satellite Beam Hopping Training')
    
    # 实验设置
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of training episodes (increased)')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save interval')
    
    # 环境参数
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams')
    parser.add_argument('--num_cells', type=int, default=19, help='Number of cells')
    parser.add_argument('--satellite_height', type=float, default=550.0, help='Satellite height (km)')
    parser.add_argument('--total_power', type=float, default=39.0, help='Total power (dBW)')
    parser.add_argument('--total_bandwidth', type=float, default=200.0, help='Total bandwidth (MHz)')
    
    # 队列参数
    parser.add_argument('--max_queue_size', type=int, default=100, help='Maximum queue size')
    parser.add_argument('--traffic_intensity', type=float, default=0.3, help='Maximum traffic_intensity')
    parser.add_argument('--packet_ttl', type=int, default=20, help='Packet time-to-live (slots)')
    parser.add_argument('--packet_size', type=float, default=100.0, help='Packet size (KB)')
    
    # 优化后的强化学习参数
    parser.add_argument('--lr_actor', type=float, default=1e-3, help='Actor learning rate (increased)')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate (increased)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--entropy_coef', type=float, default=0.05, help='Entropy coefficient (increased for exploration)')
    parser.add_argument('--update_epochs', type=int, default=15, help='Number of update epochs (increased)')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini-batch size')
    
    # 改进的奖励权重
    parser.add_argument('--reward_throughput_weight', type=float, default=1.0, help='Throughput reward weight')
    parser.add_argument('--reward_delay_weight', type=float, default=0.3, help='Delay reward weight')
    parser.add_argument('--reward_fairness_weight', type=float, default=0.2, help='Fairness reward weight')
    
    # 课程学习参数
    parser.add_argument('--curriculum_learning', action='store_true', default=True, help='Enable curriculum learning')
    parser.add_argument('--curriculum_stages', type=int, default=4, help='Number of curriculum stages')
    
    # 数据参数
    parser.add_argument('--generate_data', action='store_true', default=False, help='Generate new data')
    
    # 输出设置
    parser.add_argument('--data_dir', type=str, default='data_dir', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='improved_results', help='Output directory')
    parser.add_argument('--model_dir', type=str, default='improved_models', help='Model directory')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    return parser.parse_args()


class CurriculumLearning:
    """课程学习类"""
    
    def __init__(self, stages=4, episodes_per_stage=500):
        self.stages = stages
        self.episodes_per_stage = episodes_per_stage
        self.current_stage = 0
        
    def get_stage(self, episode):
        """获取当前阶段"""
        stage = min(episode // self.episodes_per_stage, self.stages - 1)
        return stage
    
    def get_difficulty_params(self, stage):
        """获取当前阶段的难度参数"""
        # 随着阶段增加，逐渐增加难度
        traffic_intensity = 0.3 + 0.2 * stage  # 流量强度从0.3增加到0.9
        max_queue_size = 100 + 25 * stage  # 队列大小从100增加到175
        packet_ttl = 20 - 2 * stage  # TTL从20减少到12
        
        return {
            'traffic_intensity': min(traffic_intensity, 1.0),
            'max_queue_size': min(max_queue_size, 200),
            'packet_ttl': max(packet_ttl, 10)
        }


def setup_environment(args, difficulty_params=None):
    """
    设置改进的实验环境
    
    Args:
        args: 参数对象
        difficulty_params: 课程学习难度参数
    
    Returns:
        env: 环境对象
    """
    # 使用课程学习参数或默认参数
    if difficulty_params:
        max_queue_size = difficulty_params['max_queue_size']
        packet_ttl = difficulty_params['packet_ttl']
        traffic_intensity = difficulty_params['traffic_intensity']
    else:
        max_queue_size = args.max_queue_size
        packet_ttl = args.packet_ttl
        traffic_intensity = args.traffic_intensity  # 默认流量强度
    
    # 创建环境
    env = SatelliteEnv(
        num_beams=args.num_beams,
        num_cells=args.num_cells,
        total_power_dbw=args.total_power,
        total_bandwidth_mhz=args.total_bandwidth,
        satellite_height_km=args.satellite_height,
        max_queue_size=max_queue_size,
        packet_ttl=packet_ttl,
        reward_throughput_weight=args.reward_throughput_weight,
        reward_delay_weight=args.reward_delay_weight,
        traffic_intensity=traffic_intensity
    )
    
    # 设置最大步数
    env.max_steps = args.max_steps
    
    return env


def setup_improved_agent(env, args):
    """
    设置改进的强化学习代理
    
    Args:
        env: 环境对象
        args: 参数对象
    
    Returns:
        agent: 代理对象
    """
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    discrete_action_dim = args.num_beams
    continuous_action_dim = args.num_beams
    discrete_action_space_size = args.num_cells
    
    # 创建改进的PPO代理
    agent = PPO(
        state_dim=state_dim,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_action_space_size=discrete_action_space_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,  # 增加探索
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return agent


def improved_train(env, agent, args):
    """
    改进的训练函数
    
    Args:
        env: 环境对象
        agent: 代理对象
        args: 参数对象
    
    Returns:
        metrics: 训练指标字典
    """
    # 创建模型目录
    os.makedirs(args.model_dir, exist_ok=True)
      # 初始化课程学习
    if args.curriculum_learning:
        curriculum = CurriculumLearning(stages=args.curriculum_stages, 
                                      episodes_per_stage=args.num_episodes // args.curriculum_stages)
    
    # 初始化指标
    metrics = {
        'rewards': [],
        'episode_rewards': [],
        'throughputs': [],
        'delays': [],
        'packet_losses': [],
        'fairness': [],
        'spectral_efficiency': [],
        'power_efficiency': [],
        'beam_allocation_counts': np.zeros((args.num_beams, args.num_cells)),
        'power_allocations': [],
        'sinrs': [],
        'queue_lengths': [],
        'curriculum_stages': []
    }
    
    # 训练循环
    best_reward = -float('inf')
    
    for episode in tqdm(range(args.num_episodes), desc='Improved Training'):        # 课程学习：调整环境难度
        if args.curriculum_learning:
            stage = curriculum.get_stage(episode)
            if stage != curriculum.current_stage:
                print(f"\n=== 进入课程学习阶段 {stage + 1}/{args.curriculum_stages} ===")
                curriculum.current_stage = stage
                difficulty_params = curriculum.get_difficulty_params(stage)
                
                # 动态调整环境参数而不是重新创建环境
                print(f"调整环境参数: {difficulty_params}")
                env.set_traffic_intensity(difficulty_params['traffic_intensity'])
                # 注意：max_queue_size 和 packet_ttl 需要在环境初始化时设定，不能动态修改
                
            metrics['curriculum_stages'].append(stage)
        else:
            metrics['curriculum_stages'].append(0)
        
        # 重置环境
        state = env.reset()
        state = state.flatten()
        
        episode_reward = 0
        episode_throughput = []
        episode_delays = []
        episode_packet_losses = []
        episode_fairness = []
        episode_spectral_efficiency = []
        episode_power_efficiency = []
        episode_queue_lengths = []
          # 运行一个episode
        for step in range(args.max_steps):
            # 选择动作
            action, discrete_log_prob, continuous_log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 新的奖励函数已经在 satellite_env.py 中实现
            # 直接使用环境返回的奖励值
            
            # 存储经验
            next_state_flat = next_state.flatten()
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
                episode_fairness.append(Metrics.calculate_fairness_index(info['data_rates']))
                episode_spectral_efficiency.append(Metrics.calculate_spectral_efficiency(info['data_rates'], args.total_bandwidth))
                episode_power_efficiency.append(Metrics.calculate_power_efficiency(info['data_rates'], args.total_power))
            
            if 'delay' in info:
                episode_delays.append(info['delay'])
            if 'packet_loss' in info:
                episode_packet_losses.append(info['packet_loss'])
            if 'queue_lengths' in info:
                episode_queue_lengths.extend(info['queue_lengths'])
            
            state = next_state_flat
            
            if done:
                break
        
        # 更新网络
        if len(agent.buffer) >= args.mini_batch_size:
            agent.update()
        
        # 记录episode指标
        metrics['episode_rewards'].append(episode_reward)
        metrics['rewards'].append(episode_reward)
        metrics['throughputs'].append(np.mean(episode_throughput) if episode_throughput else 0)
        metrics['delays'].append(np.mean(episode_delays) if episode_delays else 0)
        metrics['packet_losses'].append(np.mean(episode_packet_losses) if episode_packet_losses else 0)
        metrics['fairness'].append(np.mean(episode_fairness) if episode_fairness else 0)
        metrics['spectral_efficiency'].append(np.mean(episode_spectral_efficiency) if episode_spectral_efficiency else 0)
        metrics['power_efficiency'].append(np.mean(episode_power_efficiency) if episode_power_efficiency else 0)
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.model_dir, 'ppo_model_best.pt'))
        
        # 定期保存模型
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(args.model_dir, f'ppo_model_episode_{episode + 1}.pt'))
        
        # 定期评估
        if (episode + 1) % args.eval_interval == 0 and args.verbose:
            avg_reward = np.mean(metrics['episode_rewards'][-args.eval_interval:])
            avg_throughput = np.mean(metrics['throughputs'][-args.eval_interval:])
            avg_delay = np.mean(metrics['delays'][-args.eval_interval:])
            print(f"\nEpisode {episode + 1}/{args.num_episodes}:")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Throughput: {avg_throughput:.4f}")
            print(f"  Avg Delay: {avg_delay:.4f}")
            if args.curriculum_learning:
                print(f"  Curriculum Stage: {stage + 1}/{args.curriculum_stages}")
    
    # 保存最终模型
    agent.save(os.path.join(args.model_dir, 'ppo_model_final_improved.pt'))
    
    return metrics


def main():
    """
    改进的主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=== 改进的卫星波束跳跃训练 ===")
    
    # 设置环境
    print("设置环境...")
    env = setup_environment(args)
    
    # 设置代理
    print("设置改进的PPO代理...")
    agent = setup_improved_agent(env, args)
    
    # 训练代理
    print("开始改进训练...")
    start_time = time.time()
    metrics = improved_train(env, agent, args)
    training_time = time.time() - start_time
    print(f"训练完成！用时 {training_time:.2f} 秒")
    
    # 保存训练结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制改进的学习曲线
    plt.figure(figsize=(15, 10))
    
    # 奖励曲线
    plt.subplot(2, 3, 1)
    plt.plot(metrics['episode_rewards'])
    plt.plot(np.convolve(metrics['episode_rewards'], np.ones(50)/50, mode='valid'))
    plt.title('Improved Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Raw', 'Moving Average (50)'])
    
    # 吞吐量曲线
    plt.subplot(2, 3, 2)
    plt.plot(metrics['throughputs'])
    plt.title('Throughput Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    
    # 延迟曲线
    plt.subplot(2, 3, 3)
    plt.plot(metrics['delays'])
    plt.title('Delay Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Delay')
    
    # 公平性曲线
    plt.subplot(2, 3, 4)
    plt.plot(metrics['fairness'])
    plt.title('Fairness Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Fairness Index')
    
    # 频谱效率曲线
    plt.subplot(2, 3, 5)
    plt.plot(metrics['spectral_efficiency'])
    plt.title('Spectral Efficiency Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Spectral Efficiency')
    
    # 课程学习阶段
    if args.curriculum_learning:
        plt.subplot(2, 3, 6)
        plt.plot(metrics['curriculum_stages'])
        plt.title('Curriculum Learning Stages')
        plt.xlabel('Episode')
        plt.ylabel('Stage')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'improved_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("改进训练完成！")
    print(f"最佳模型保存在: {os.path.join(args.model_dir, 'ppo_model_best.pt')}")
    print(f"最终模型保存在: {os.path.join(args.model_dir, 'ppo_model_final_improved.pt')}")


if __name__ == "__main__":
    main()
