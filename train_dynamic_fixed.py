"""
修复版动态环境训练脚本
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.dynamic_satellite_env import DynamicSatelliteEnv
from rl.ppo import PPO
from algorithms.baseline_algorithms import GreedyAlgorithm, RandomAlgorithm

def create_dynamic_env(complexity='medium'):
    """创建动态环境"""
    if complexity == 'simple':
        return DynamicSatelliteEnv(
            num_beams=4, num_cells=19,
            enable_channel_dynamics=False,
            enable_traffic_dynamics=False,
            enable_weather=False,
            enable_mobility=False
        )
    elif complexity == 'medium':
        return DynamicSatelliteEnv(
            num_beams=4, num_cells=19,
            enable_channel_dynamics=True,
            enable_traffic_dynamics=True,
            enable_weather=False,
            enable_mobility=False
        )
    else:  # complex
        return DynamicSatelliteEnv(
            num_beams=4, num_cells=19,
            enable_channel_dynamics=True,
            enable_traffic_dynamics=True,
            enable_weather=True,
            enable_mobility=True
        )

def train_ppo_fixed(complexity='medium', episodes=200):
    """修复版PPO训练"""
    print(f"在{complexity}复杂度环境中训练PPO...")
    
    # 创建环境
    env = create_dynamic_env(complexity)
    
    # 创建PPO智能体
    obs_dim = env.observation_space.shape[0]
    
    agent = PPO(
        state_dim=obs_dim,
        discrete_action_dim=env.num_beams,  # 4个波束
        continuous_action_dim=env.num_beams,  # 4个功率分配
        discrete_action_space_size=env.num_cells,  # 19个小区
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01
    )
    
    # 训练统计
    episode_rewards = []
    episode_throughputs = []
    
    best_reward = -float('inf')
    
    print(f"开始训练，总共{episodes}个episode...")
    
    for episode in tqdm(range(episodes), desc=f"训练({complexity})"):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_info = []
        
        for step in range(200):  # 限制每个episode的步数
            # 选择动作
            action, discrete_log_prob, continuous_log_prob, state_value = agent.select_action(obs)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 从action字典中提取离散和连续动作
            discrete_action = action['beam_allocation']
            continuous_action = action['power_allocation']
              # 存储经验
            agent.buffer.add(
                obs, discrete_action, continuous_action, 
                discrete_log_prob, continuous_log_prob, reward, state_value, done
            )
            
            episode_reward += reward
            episode_info.append(info)
            
            obs = next_obs
            episode_steps += 1
            
            if done:
                break
          # 更新智能体
        if len(agent.buffer) >= agent.mini_batch_size:
            agent.update()
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        avg_throughput = np.mean([info['throughput'] for info in episode_info])
        episode_throughputs.append(avg_throughput)
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs('dynamic_models', exist_ok=True)
            model_path = f'dynamic_models/ppo_best_{complexity}.pt'
            agent.save(model_path)
        
        # 定期打印统计
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_throughputs = episode_throughputs[-50:]
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  平均奖励: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}")
            print(f"  平均吞吐量: {np.mean(recent_throughputs):.3f}")
            print(f"  最佳奖励: {best_reward:.3f}")
    
    # 保存最终模型
    final_model_path = f'dynamic_models/ppo_final_{complexity}.pt'
    agent.save(final_model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title(f'训练奖励 ({complexity})')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_throughputs)
    plt.title(f'吞吐量 ({complexity})')
    plt.xlabel('Episode')
    plt.ylabel('吞吐量')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('dynamic_results', exist_ok=True)
    plt.savefig(f'dynamic_results/training_curves_{complexity}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return agent, {
        'rewards': episode_rewards,
        'throughputs': episode_throughputs
    }

def evaluate_algorithm_fixed(env, algorithm, num_episodes=10):
    """修复版算法评估"""
    total_rewards = []
    total_throughputs = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_throughput = 0
        
        for step in range(100):  # 限制步数
            if hasattr(algorithm, 'select_action'):
                # PPO智能体
                action, _, _, _ = algorithm.select_action(obs)
            else:
                # 基线算法
                action = algorithm.get_action(obs, env)
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_throughput += info['throughput']
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_throughputs.append(episode_throughput)
    
    return {
        'reward': np.mean(total_rewards),
        'reward_std': np.std(total_rewards),
        'throughput': np.mean(total_throughputs),
        'throughput_std': np.std(total_throughputs)
    }

def compare_algorithms_fixed(complexity='medium'):
    """修复版算法比较"""
    print(f"在{complexity}复杂度环境中比较算法性能...")
    
    env = create_dynamic_env(complexity)
    results = {}
    
    # 评估贪心算法
    print("评估贪心算法...")
    greedy = GreedyAlgorithm()
    results['贪心'] = evaluate_algorithm_fixed(env, greedy, 10)
    
    # 评估随机算法
    print("评估随机算法...")
    random_alg = RandomAlgorithm(num_beams=4, num_cells=19)
    results['随机'] = evaluate_algorithm_fixed(env, random_alg, 10)
    
    # 评估PPO（如果已训练）
    model_path = f'dynamic_models/ppo_best_{complexity}.pt'
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
        results['PPO'] = evaluate_algorithm_fixed(env, ppo_agent, 10)
    else:
        print(f"PPO模型不存在: {model_path}")
    
    # 打印结果
    print(f"\n{complexity}复杂度环境下的性能比较:")
    print("-" * 60)
    for alg_name, result in results.items():
        print(f"{alg_name:>6}: 奖励={result['reward']:.3f}±{result['reward_std']:.3f}, "
              f"吞吐量={result['throughput']:.3f}±{result['throughput_std']:.3f}")
    
    return results

def main():
    """主函数"""
    print("动态卫星环境PPO训练测试 (修复版)")
    print("=" * 50)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 选择复杂度
    complexity = 'medium'  # 固定使用medium复杂度
    
    print(f"使用{complexity}复杂度环境")
    
    # 训练PPO
    print("\n开始训练PPO...")
    agent, training_stats = train_ppo_fixed(complexity, episodes=100)  # 减少episode数量以快速测试
    
    # 比较算法性能
    print("\n比较算法性能...")
    comparison_results = compare_algorithms_fixed(complexity)
    
    print("\n训练和评估完成！")
    print(f"模型保存在: dynamic_models/ppo_best_{complexity}.pt")
    print(f"结果图保存在: dynamic_results/training_curves_{complexity}.png")

if __name__ == "__main__":
    main()
