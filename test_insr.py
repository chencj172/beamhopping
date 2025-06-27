import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

from environment.dynamic_satellite_env import DynamicSatelliteEnv
from rl.ppo import PPO
from algorithms.baseline_algorithms import GreedyAlgorithm, RandomAlgorithm

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_env(complexity_level='complex', traffic_intensity=0.8):
    """创建指定复杂度的环境"""
    if complexity_level == 'simple':
        env = DynamicSatelliteEnv(
            num_beams=4,
            num_cells=19,
            enable_channel_dynamics=False,
            enable_traffic_dynamics=False,
            enable_weather=False,
            enable_mobility=False,
            traffic_intensity=traffic_intensity
        )
    elif complexity_level == 'medium':
        env = DynamicSatelliteEnv(
            num_beams=4,
            num_cells=19,
            enable_channel_dynamics=True,
            enable_traffic_dynamics=True,
            enable_weather=False,
            enable_mobility=False,
            traffic_intensity=traffic_intensity
        )
    else:  # complex
        env = DynamicSatelliteEnv(
            num_beams=3,
            num_cells=19,
            enable_channel_dynamics=True,
            enable_traffic_dynamics=True,
            enable_weather=True,
            enable_mobility=True,
            traffic_intensity=traffic_intensity
        )
    
    return env


def evaluate_algorithm(env, algorithm, num_episodes=10):
    """评估算法性能"""
    total_rewards = []
    total_throughputs = []
    total_delays = []
    total_fairness = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_throughput = 0
        episode_delays = []
        episode_fairness = []
        done = False
        step_count = 0
        while not done and step_count < 1000:  # 限制最大步数
            if hasattr(algorithm, 'select_action'):
                # PPO智能体
                action, _, _, _ = algorithm.select_action(obs)
            else:
                # 基线算法
                action = algorithm.get_action(obs, env)
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_throughput += info['throughput']
            episode_delays.append(info['delay'])
            if 'fairness_index' in info:
                episode_fairness.append(info['fairness_index'])
            
            step_count += 1
        
        total_rewards.append(episode_reward)
        total_throughputs.append(episode_throughput)
        total_delays.append(np.mean(episode_delays) if episode_delays else 0)
        total_fairness.append(np.mean(episode_fairness) if episode_fairness else 1.0)
    
    return {
        'reward': np.mean(total_rewards),
        'reward_std': np.std(total_rewards),
        'throughput': np.mean(total_throughputs),
        'throughput_std': np.std(total_throughputs),
        'delay': np.mean(total_delays),
        'delay_std': np.std(total_delays),
        'fairness': np.mean(total_fairness),
        'fairness_std': np.std(total_fairness)
    }

def plot_algorithm_performance(results, complexity_level='complex'):
    """
    绘制算法性能对比图
    
    Args:
        results: 评估结果字典
        complexity_level: 复杂度等级
    """
    if not results:
        print("没有评估结果可以绘制")
        return
    
    algorithms = list(results.keys())
    metrics = ['reward', 'throughput', 'delay', 'fairness']
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{complexity_level.upper()}环境下算法性能对比', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # 准备数据
        values = []
        errors = []
        labels = []
        
        for alg in algorithms:
            if metric in results[alg]:
                values.append(results[alg][metric])
                errors.append(results[alg].get(f'{metric}_std', 0))
                labels.append(alg)
        
        if not values:
            continue
            
        # 绘制柱状图
        bars = ax.bar(labels, values, yerr=errors, capsize=5, 
                     color=colors[:len(labels)], alpha=0.8)
        
        # 在柱子上添加数值标签
        for j, (bar, value, error) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} compare')
        ax.grid(True, alpha=0.3)
        
        # 设置Y轴范围
        if metric == 'fairness':
            ax.set_ylim(0, 1.1)
        elif metric == 'delay':
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    plt.show()

print("评估贪心算法...")
# 创建环境
env = create_env()

results = {}
greedy = GreedyAlgorithm()
results['Greedy'] = evaluate_algorithm(env, greedy, 100)
plot_algorithm_performance(results)

