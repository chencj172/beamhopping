"""
动态环境训练脚本
测试新的动态卫星环境，对比不同复杂度下PPO的表现
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
from algorithms.baseline_algorithms import GreedyAlgorithm, RandomAlgorithm

def create_env(complexity_level='complex', traffic_intensity=0.7):
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
            num_beams=4,
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

def train_ppo_dynamic(complexity_level='complex', episodes=1000, traffic_intensity=0.7):
    """在动态环境中训练PPO"""
    print(f"开始在{complexity_level}复杂度环境中训练PPO...")
      # 创建环境
    env = create_env(complexity_level, traffic_intensity)
    
    # 创建PPO智能体
    obs_dim = env.observation_space.shape[0]
    
    agent = PPO(
        state_dim=obs_dim,
        discrete_action_dim=env.num_beams,  # beam allocation
        continuous_action_dim=env.num_beams,  # power allocation  
        discrete_action_space_size=env.num_cells,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01,
        update_epochs=4
    )
      # 训练统计
    episode_rewards = []
    episode_throughputs = []
    episode_delays = []
    episode_fairness = []
    episode_complexity = []
    
    best_reward = -float('inf')
    
    for episode in tqdm(range(episodes), desc=f"训练 ({complexity_level})"):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_info = []
        
        # 每50个episode打印一次详细的初始环境状态
        if (episode + 1) % 50 == 0:
            print(f"\n--- Episode {episode + 1} 初始环境状态 ---")
            print(f"天气严重性: {env.env_state.get('weather_severity', 0):.3f}")
            print(f"流量突发性: {env.env_state.get('traffic_burstiness', 0):.3f}")
            print(f"信道质量: {env.env_state.get('channel_quality', 1):.3f}")
            
            # 显示前5个小区的详细信息
            current_traffic = env.queue_model.get_current_traffic()
            current_channels = env.channel_model.get_current_channel_state()
            print("前5个小区状态:")
            for i in range(min(5, len(current_traffic))):
                print(f"  小区{i}: 业务请求={current_traffic[i]:.2f}, 信道质量={current_channels[i]:.3f}")
            print("=" * 40)
        
        while episode_steps < 1000:  # 限制每个episode的最大步数
            # 选择动作
            action, discrete_log_prob, continuous_log_prob, state_value = agent.select_action(obs)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 存储经验
            agent.buffer.add(
                obs,
                action['beam_allocation'], 
                action['power_allocation'],
                discrete_log_prob,
                continuous_log_prob,
                reward,
                state_value,
                done
            )
            
            episode_reward += reward
            episode_info.append(info)
            
            obs = next_obs
            episode_steps += 1

            # if(episode_steps % 100 == 0):
            #     # 打印当前环境动态信息
            #     print("  当前环境状态:")
            #     print(f"    天气严重性: {env.env_state.get('weather_severity', 0):.3f}")
            #     print(f"    流量突发性: {env.env_state.get('traffic_burstiness', 0):.3f}")
            #     print(f"    信道质量: {env.env_state.get('channel_quality', 1):.3f}")
                
            #     # 打印小区业务请求量
            #     current_traffic = env.queue_model.get_current_traffic()
            #     print(f"    小区业务请求量 (前10个): {current_traffic[:10]}")
            #     print(f"    业务请求量统计: 最小={np.min(current_traffic):.2f}, 最大={np.max(current_traffic):.2f}, 平均={np.mean(current_traffic):.2f}")
            
            #     # 打印信道条件
            #     current_channels = env.channel_model.get_current_channel_state()
            #     print(f"    信道条件 (前10个): {current_channels[:10]}")
            
            if done:
                break        # 更新智能体
        if len(agent.buffer) >= agent.mini_batch_size:
            agent.update()
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        avg_throughput = np.mean([info['throughput'] for info in episode_info])
        avg_delay = np.mean([info['delay'] for info in episode_info])
        avg_fairness = np.mean([info.get('fairness_index', 1.0) for info in episode_info])
        complexity_score = env.get_environment_complexity_score()
        
        episode_throughputs.append(avg_throughput)
        episode_delays.append(avg_delay)
        episode_fairness.append(avg_fairness)
        episode_complexity.append(complexity_score)
        

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = f'dynamic_models/ppo_best_{complexity_level}.pt'
            os.makedirs('dynamic_models', exist_ok=True)
            agent.save(model_path)
          # 定期保存模型和打印统计
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  平均奖励: {np.mean(episode_rewards[-100:]):.3f}")
            print(f"  平均吞吐量: {np.mean(episode_throughputs[-100:]):.3f}")
            print(f"  平均延迟: {np.mean(episode_delays[-100:]):.3f}")
            print(f"  平均公平性: {np.mean(episode_fairness[-100:]):.3f}")
            print(f"  环境复杂度: {complexity_score:.3f}")
            
            # 保存周期性模型
            model_path = f'dynamic_models/ppo_{complexity_level}_episode_{episode + 1}.pt'
            agent.save(model_path)
    
    # 保存最终模型
    final_model_path = f'dynamic_models/ppo_final_{complexity_level}.pt'
    agent.save(final_model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards)
    plt.title(f'train reword ({complexity_level})')
    plt.xlabel('Episode')
    plt.ylabel('reword')
    
    plt.subplot(2, 3, 2)
    plt.plot(episode_throughputs)
    plt.title(f'throughput ({complexity_level})')
    plt.xlabel('Episode')
    plt.ylabel('throughput')
    
    plt.subplot(2, 3, 3)
    plt.plot(episode_delays)
    plt.title(f'delay ({complexity_level})')
    plt.xlabel('Episode')
    plt.ylabel('delay')
    
    plt.subplot(2, 3, 4)
    plt.plot(episode_fairness)
    plt.title(f'fairness ({complexity_level})')
    plt.xlabel('Episode')
    plt.ylabel('fairness index')
    
    plt.subplot(2, 3, 5)
    plt.plot(episode_complexity)
    plt.title(f'environment complexity ({complexity_level})')
    plt.xlabel('Episode')
    plt.ylabel('complexity score')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('dynamic_results', exist_ok=True)
    plt.savefig(f'dynamic_results/training_curves_{complexity_level}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return agent, {
        'rewards': episode_rewards,
        'throughputs': episode_throughputs,
        'delays': episode_delays,
        'fairness': episode_fairness,
        'complexity': episode_complexity
    }

def evaluate_all_algorithms(complexity_level='complex', num_episodes=20, traffic_intensity=0.7):
    """评估所有算法在指定复杂度环境下的性能"""
    print(f"在{complexity_level}复杂度环境中评估所有算法...")
    
    # 创建环境
    env = create_env(complexity_level, traffic_intensity)
    
    results = {}
    
    # 评估贪心算法
    print("评估贪心算法...")
    greedy = GreedyAlgorithm()
    results['Greedy'] = evaluate_algorithm(env, greedy, num_episodes)
    
    # 评估随机算法
    print("评估随机算法...")
    random_alg = RandomAlgorithm(num_beams=env.num_beams, num_cells=env.num_cells)
    results['Random'] = evaluate_algorithm(env, random_alg, num_episodes)
      # 评估PPO（如果已训练）
    model_path = f'dynamic_models/ppo_best_{complexity_level}.pt'
    if os.path.exists(model_path):
        print("评估PPO...")
        obs_dim = env.observation_space.shape[0]
        
        ppo_agent = PPO(
            state_dim=obs_dim,
            discrete_action_dim=env.num_beams,
            continuous_action_dim=env.num_beams,
            discrete_action_space_size=env.num_cells
        )
        ppo_agent.load(model_path)
        results['PPO'] = evaluate_algorithm(env, ppo_agent, num_episodes)
    else:
        print(f"PPO模型不存在: {model_path}")
    
    return results

def compare_complexity_levels():
    """比较不同复杂度等级下的算法性能"""
    complexity_levels = ['simple', 'medium', 'complex']
    algorithms = ['Greedy', 'Random', 'PPO']
    metrics = ['reward', 'throughput', 'delay', 'fairness']
    
    # 存储结果
    comparison_results = {level: {} for level in complexity_levels}
    
    # 评估每个复杂度等级
    for level in complexity_levels:
        print(f"\n{'='*50}")
        print(f"评估复杂度等级: {level.upper()}")
        print(f"{'='*50}")
        
        results = evaluate_all_algorithms(level, num_episodes=10)
        comparison_results[level] = results
    
    # 可视化比较结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('不同复杂度环境下的算法性能比较', fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        # 准备数据
        data = {alg: [] for alg in algorithms}
        errors = {alg: [] for alg in algorithms}
        
        for level in complexity_levels:
            for alg in algorithms:
                if alg in comparison_results[level]:
                    data[alg].append(comparison_results[level][alg][metric])
                    errors[alg].append(comparison_results[level][alg].get(f'{metric}_std', 0))
                else:
                    data[alg].append(0)
                    errors[alg].append(0)
        
        # 绘制柱状图
        x = np.arange(len(complexity_levels))
        width = 0.25
        
        for j, alg in enumerate(algorithms):
            if any(data[alg]):  # 只绘制有数据的算法
                ax.bar(x + j * width, data[alg], width, 
                      label=alg, yerr=errors[alg], capsize=5)
        
        ax.set_xlabel('环境复杂度')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} 比较')
        ax.set_xticks(x + width)
        ax.set_xticklabels([level.title() for level in complexity_levels])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_results/complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印数值结果
    print(f"\n{'='*60}")
    print("数值比较结果")
    print(f"{'='*60}")
    
    for level in complexity_levels:
        print(f"\n{level.upper()} 复杂度:")
        print("-" * 40)
        for alg in algorithms:
            if alg in comparison_results[level]:
                result = comparison_results[level][alg]
                print(f"{alg:>8}: 奖励={result['reward']:.3f}±{result['reward_std']:.3f}, "
                      f"吞吐量={result['throughput']:.3f}±{result['throughput_std']:.3f}, "
                      f"延迟={result['delay']:.3f}±{result['delay_std']:.3f}, "
                      f"公平性={result['fairness']:.3f}±{result['fairness_std']:.3f}")
    
    return comparison_results

def main():
    """主函数"""
    print("动态卫星环境训练与评估")
    print("=" * 50)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建输出目录
    os.makedirs('dynamic_models', exist_ok=True)
    os.makedirs('dynamic_results', exist_ok=True)
    
    # 选择要执行的任务
    print("请选择要执行的任务:")
    print("1. 训练PPO（简单环境）")
    print("2. 训练PPO（中等复杂度环境）")
    print("3. 训练PPO（复杂环境）")
    print("4. 评估所有算法（复杂环境）")
    print("5. 比较不同复杂度等级")
    print("6. 全部执行")
    
    choice = input("请输入选择 (1-6): ").strip()
    
    if choice == '1':
        train_ppo_dynamic('simple', episodes=500)
    elif choice == '2':
        train_ppo_dynamic('medium', episodes=800)
    elif choice == '3':
        train_ppo_dynamic('complex', episodes=1000)
    elif choice == '4':
        results = evaluate_all_algorithms('complex', num_episodes=20)
        print("评估结果:", results)
    elif choice == '5':
        compare_complexity_levels()
    elif choice == '6':
        # 执行所有任务
        print("开始全面评估...")
        
        # 训练不同复杂度的PPO
        for level in ['simple', 'medium', 'complex']:
            episodes = {'simple': 500, 'medium': 800, 'complex': 1000}[level]
            train_ppo_dynamic(level, episodes=episodes)
        
        # 比较复杂度等级
        compare_complexity_levels()
        
        print("全面评估完成！")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
