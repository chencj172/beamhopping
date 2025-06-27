"""
分析PPO算法的性能问题
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.dynamic_satellite_env import DynamicSatelliteEnv
from rl.ppo import PPO
from algorithms.baseline_algorithms import GreedyAlgorithm

def analyze_ppo_performance():
    """分析PPO性能问题"""
    print("开始分析PPO性能问题...")
    
    # 创建复杂环境
    env = DynamicSatelliteEnv(
        num_beams=3,
        num_cells=19,
        enable_channel_dynamics=True,
        enable_traffic_dynamics=True,
        enable_weather=True,
        enable_mobility=True,
        traffic_intensity=0.8
    )
    
    # 加载PPO模型
    model_path = 'dynamic_models/ppo_best_complex.pt'
    if not os.path.exists(model_path):
        print(f"PPO模型不存在: {model_path}")
        return
    
    obs_dim = env.observation_space.shape[0]
    ppo_agent = PPO(
        state_dim=obs_dim,
        discrete_action_dim=env.num_beams,
        continuous_action_dim=env.num_beams,
        discrete_action_space_size=env.num_cells
    )
    ppo_agent.load(model_path)
    
    # 创建贪心算法
    greedy = GreedyAlgorithm()
    
    # 运行分析
    print("\n比较PPO和Greedy算法的详细行为...")
    
    algorithms = [('PPO', ppo_agent), ('Greedy', greedy)]
    results = {}
    
    for alg_name, algorithm in algorithms:
        print(f"\n分析{alg_name}算法...")
        
        obs = env.reset()
        episode_rewards = []
        episode_delays = []
        episode_throughputs = []
        episode_served = []
        episode_queue_lengths = []
        
        for step in range(50):  # 运行50步分析
            if hasattr(algorithm, 'select_action'):
                # PPO
                action, _, _, _ = algorithm.select_action(obs)
            else:
                # Greedy
                action = algorithm.get_action(obs, env)
            
            obs, reward, done, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_delays.append(info['delay'])
            episode_throughputs.append(info['throughput'])
            episode_served.append(np.sum(info['served_per_cell']))
            episode_queue_lengths.append(np.sum(env.queue_model.get_queue_lengths()))
            
            if step < 10:  # 打印前10步的详细信息
                print(f"  步骤{step+1}: 奖励={reward:.3f}, 延迟={info['delay']:.3f}, "
                      f"吞吐量={info['throughput']:.3f}, 服务数据包={np.sum(info['served_per_cell']):.0f}, "
                      f"队列总长度={np.sum(env.queue_model.get_queue_lengths()):.0f}")
        
        results[alg_name] = {
            'rewards': episode_rewards,
            'delays': episode_delays,
            'throughputs': episode_throughputs,
            'served': episode_served,
            'queue_lengths': episode_queue_lengths
        }
        
        print(f"  {alg_name}平均性能:")
        print(f"    平均奖励: {np.mean(episode_rewards):.3f}")
        print(f"    平均延迟: {np.mean(episode_delays):.3f}")
        print(f"    平均吞吐量: {np.mean(episode_throughputs):.3f}")
        print(f"    平均服务数据包: {np.mean(episode_served):.3f}")
        print(f"    平均队列长度: {np.mean(episode_queue_lengths):.3f}")
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PPO vs Greedy Detailed Performance Analysis', fontsize=16)
    
    metrics = ['rewards', 'delays', 'throughputs', 'served', 'queue_lengths']
    titles = ['Rewards', 'Delays', 'Throughputs', 'Served Packets', 'Queue Lengths']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if i < 5:
            ax = axes[i // 3, i % 3]
            
            for alg_name in ['PPO', 'Greedy']:
                ax.plot(results[alg_name][metric], label=alg_name, marker='o', markersize=3)
            
            ax.set_title(title)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 最后一个子图：奖励成分分析
    ax = axes[1, 2]
    ax.text(0.1, 0.8, 'Reward Analysis', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.6, f'PPO avg reward: {np.mean(results["PPO"]["rewards"]):.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.5, f'Greedy avg reward: {np.mean(results["Greedy"]["rewards"]):.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.3, f'Delay difference: {np.mean(results["PPO"]["delays"]) - np.mean(results["Greedy"]["delays"]):.3f}', transform=ax.transAxes)
    ax.text(0.1, 0.2, f'Throughput difference: {np.mean(results["PPO"]["throughputs"]) - np.mean(results["Greedy"]["throughputs"]):.3f}', transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig('analysis_results/ppo_vs_greedy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析奖励函数权重的影响
    print(f"\n当前奖励权重: {env.reward_weights}")
    print("\n建议的优化方向:")
    print("1. 增加延迟权重，减少能效权重")
    print("2. 调整奖励函数，使延迟惩罚更明显") 
    print("3. 增加训练轮数，特别是在复杂环境下")
    print("4. 考虑使用经验回放或其他稳定化技术")

if __name__ == "__main__":
    analyze_ppo_performance()
