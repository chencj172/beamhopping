"""
简化版模型评估脚本

直接运行对比实验，无需命令行参数
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict
import os

from environment.satellite_env import SatelliteEnv
from rl.ppo import PPO

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def greedy_allocation(traffic_demand, num_beams, num_cells, total_power=1.0):
    """贪心算法：根据请求量分配波束"""
    # 选择请求量最大的小区
    sorted_cells = np.argsort(traffic_demand)[::-1]  # 降序排列
    
    beam_allocation = []
    for i in range(num_beams):
        if i < len(sorted_cells):
            beam_allocation.append(sorted_cells[i])
        else:
            beam_allocation.append(0)  # 如果波束数量多于小区，分配给第一个小区
    
    # 功率均分
    power_allocation = np.ones(num_beams) * (total_power / num_beams)
    
    return {
        'beam_allocation': beam_allocation,
        'power_allocation': power_allocation
    }


def random_allocation(num_beams, num_cells, total_power=1.0):
    """随机算法：随机分配波束"""
    # 随机选择小区
    beam_allocation = np.random.choice(num_cells, num_beams, replace=True).tolist()
    
    # 功率均分
    power_allocation = np.ones(num_beams) * (total_power / num_beams)
    
    return {
        'beam_allocation': beam_allocation,
        'power_allocation': power_allocation
    }


def load_ppo_model(env, model_path):
    """加载PPO模型"""
    try:        # 获取环境参数
        initial_state = env.reset()
        state_dim = initial_state.shape[0] * initial_state.shape[1]  # 展平后的维度
        discrete_action_dim = env.num_beams
        continuous_action_dim = env.num_beams
        discrete_action_space_size = env.num_cells
        
        # 创建PPO智能体
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = PPO(
            state_dim=state_dim,
            discrete_action_dim=discrete_action_dim,
            continuous_action_dim=continuous_action_dim,
            discrete_action_space_size=discrete_action_space_size,
            device=device
        )
          # 加载模型参数
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'discrete_actor' in checkpoint:
            # 新格式：字典包含各个组件的state_dict
            agent.discrete_actor.load_state_dict(checkpoint['discrete_actor'])
            agent.continuous_actor.load_state_dict(checkpoint['continuous_actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            print(f"成功加载模型: {model_path}")
        elif 'discrete_actor_state_dict' in checkpoint:
            # 旧格式：直接包含state_dict
            agent.discrete_actor.load_state_dict(checkpoint['discrete_actor_state_dict'])
            agent.continuous_actor.load_state_dict(checkpoint['continuous_actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"加载模型成功 (兼容模式): {model_path}")
        else:
            # 最老格式：直接是state_dict（仅离散actor）
            agent.discrete_actor.load_state_dict(checkpoint)
            print(f"加载模型成功 (仅离散actor): {model_path}")
        
        return agent
    
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


def evaluate_algorithm(env, algorithm, agent=None, num_episodes=10, num_steps=100):
    """评估单个算法"""
    results = {
        'rewards': [],
        'throughputs': [],
        'spectral_efficiencies': [],
        'power_efficiencies': [],
        'fairness_indices': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []
        episode_throughputs = []
        episode_spectral_efficiencies = []
        episode_power_efficiencies = []
        episode_fairness_indices = []
        
        for step in range(num_steps):            # 根据算法选择动作
            if algorithm == 'ppo' and agent is not None:
                # PPO可能需要一维状态
                state_flat = state.flatten() if state.ndim > 1 else state
                action, _, _, _ = agent.select_action(state_flat)
            elif algorithm == 'greedy':
                traffic_demand = env.get_traffic_demand()
                action = greedy_allocation(traffic_demand, env.num_beams, env.num_cells)
            elif algorithm == 'random':
                action = random_allocation(env.num_beams, env.num_cells)
            else:
                raise ValueError(f"未知算法: {algorithm}")
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录指标
            episode_rewards.append(reward)
            if 'metrics' in info:
                metrics = info['metrics']
                episode_throughputs.append(metrics.get('total_throughput', 0))
                episode_spectral_efficiencies.append(metrics.get('spectral_efficiency', 0))
                episode_power_efficiencies.append(metrics.get('power_efficiency', 0))
                episode_fairness_indices.append(metrics.get('fairness_index', 0))
            
            state = next_state
            
            if done:
                break
        
        # 存储episode结果
        results['rewards'].append(np.mean(episode_rewards))
        results['throughputs'].append(np.mean(episode_throughputs) if episode_throughputs else 0)
        results['spectral_efficiencies'].append(np.mean(episode_spectral_efficiencies) if episode_spectral_efficiencies else 0)
        results['power_efficiencies'].append(np.mean(episode_power_efficiencies) if episode_power_efficiencies else 0)
        results['fairness_indices'].append(np.mean(episode_fairness_indices) if episode_fairness_indices else 0)
        
        print(f"{algorithm.upper()} Episode {episode + 1}/{num_episodes} - "
              f"Reward: {results['rewards'][-1]:.4f}, "
              f"Throughput: {results['throughputs'][-1]:.2f}, "
              f"Spectral Eff: {results['spectral_efficiencies'][-1]:.4f}")
    
    return results


def plot_comparison(all_results, save_path="evaluation_results"):
    """绘制对比结果"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    metrics = ['rewards', 'throughputs', 'spectral_efficiencies', 'power_efficiencies', 'fairness_indices']
    metric_names = ['平均奖励', '平均吞吐量', '频谱效率', '功率效率', '公平性指数']
    
    algorithms = list(all_results.keys())
    colors = {'random': 'red', 'greedy': 'blue', 'ppo': 'green'}
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 准备数据
        data = []
        labels = []
        
        for algorithm in algorithms:
            if all_results[algorithm][metric]:
                data.append(all_results[algorithm][metric])
                labels.append(algorithm.upper())
        
        # 绘制箱线图
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch, label in zip(bp['boxes'], labels):
                patch.set_facecolor(colors.get(label.lower(), 'gray'))
                patch.set_alpha(0.7)
        
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison_results.png'), dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(all_results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)
    
    algorithms = list(all_results.keys())
    metrics = ['rewards', 'throughputs', 'spectral_efficiencies', 'power_efficiencies', 'fairness_indices']
    metric_names = ['平均奖励', '平均吞吐量', '频谱效率', '功率效率', '公平性指数']
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{metric_name}:")
        print("-" * 40)
        for algorithm in algorithms:
            if all_results[algorithm][metric]:
                mean_val = np.mean(all_results[algorithm][metric])
                std_val = np.std(all_results[algorithm][metric])
                print(f"{algorithm.upper():>8}: {mean_val:8.4f} ± {std_val:8.4f}")
    
    # 计算相对性能提升
    if 'ppo' in all_results and all_results['random']['rewards']:
        print(f"\n相对性能提升 (相对于随机算法):")
        print("-" * 40)
        for algorithm in algorithms:
            if algorithm != 'random' and all_results[algorithm]['rewards']:
                algo_reward = np.mean(all_results[algorithm]['rewards'])
                random_reward = np.mean(all_results['random']['rewards'])
                if random_reward != 0:
                    improvement = (algo_reward - random_reward) / abs(random_reward) * 100
                    print(f"{algorithm.upper():>8}: {improvement:+8.2f}%")


def main():
    """主函数"""
    print("开始模型评估和对比实验...")
    
    # 创建环境
    print("创建环境...")
    env = SatelliteEnv()
    
    # 尝试加载PPO模型
    model_path = "models/ppo_model_final.pt"
    ppo_agent = None
    if os.path.exists(model_path):
        ppo_agent = load_ppo_model(env, model_path)
    else:
        print(f"模型文件不存在: {model_path}")
        print("将只运行基准算法对比")
    
    # 评估参数
    num_episodes = 5  # 减少episode数量以加快测试
    num_steps = 50    # 减少步数以加快测试
    
    # 运行评估
    all_results = {}
    
    # 评估随机算法
    print("\n评估随机算法...")
    all_results['random'] = evaluate_algorithm(env, 'random', None, num_episodes, num_steps)
    
    # 评估贪心算法
    print("\n评估贪心算法...")
    all_results['greedy'] = evaluate_algorithm(env, 'greedy', None, num_episodes, num_steps)
    
    # 评估PPO算法（如果模型加载成功）
    if ppo_agent is not None:
        print("\n评估PPO算法...")
        all_results['ppo'] = evaluate_algorithm(env, 'ppo', ppo_agent, num_episodes, num_steps)
    
    # 打印结果摘要
    print_summary(all_results)
    
    # 绘制结果
    plot_comparison(all_results)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()
