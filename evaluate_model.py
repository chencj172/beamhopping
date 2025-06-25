"""
模型验证和对比实验程序

比较三种算法的性能：
1. 训练好的PPO模型
2. 贪心算法（根据请求量分配波束）
3. 随机算法

所有算法的功率分配都采用均分策略
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import os
from collections import defaultdict

from environment.satellite_env import SatelliteEnv
from rl.ppo import PPO
from utils.metrics import Metrics

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class BaselineAlgorithms:
    """基准算法类"""
    
    def __init__(self, num_beams: int, num_cells: int, total_power: float = 1.0):
        self.num_beams = num_beams
        self.num_cells = num_cells
        self.total_power = total_power
    
    def greedy_allocation(self, traffic_demand: np.ndarray) -> Dict:
        """
        贪心算法：根据请求量分配波束
        
        Args:
            traffic_demand: 流量需求 [num_cells]
        
        Returns:
            action: 动作字典
        """
        # 选择请求量最大的小区
        sorted_cells = np.argsort(traffic_demand)[::-1]  # 降序排列
        
        beam_allocation = []
        for i in range(self.num_beams):
            if i < len(sorted_cells):
                beam_allocation.append(sorted_cells[i])
            else:
                beam_allocation.append(0)  # 如果波束数量多于小区，分配给第一个小区
        
        # 功率均分
        power_allocation = np.ones(self.num_beams) * (self.total_power / self.num_beams)
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }
    
    def random_allocation(self) -> Dict:
        """
        随机算法：随机分配波束
        
        Returns:
            action: 动作字典
        """
        # 随机选择小区
        beam_allocation = np.random.choice(self.num_cells, self.num_beams, replace=True).tolist()
        
        # 功率均分
        power_allocation = np.ones(self.num_beams) * (self.total_power / self.num_beams)
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, env: SatelliteEnv, model_path: str = None):
        self.env = env
        self.baseline = BaselineAlgorithms(env.num_beams, env.num_cells)
          # 加载训练好的PPO模型
        self.ppo_agent = None
        if model_path and os.path.exists(model_path):
            self.load_ppo_model(model_path)
        
        # 存储评估结果
        self.results = {
            'ppo': defaultdict(list),
            'greedy': defaultdict(list),
            'random': defaultdict(list)
        }
    
    def load_ppo_model(self, model_path: str):
        """加载PPO模型"""
        try:
            # 获取环境参数
            initial_state = self.env.reset()
            state_dim = initial_state.shape[0] * initial_state.shape[1]  # 展平后的维度
            discrete_action_dim = self.env.num_beams
            continuous_action_dim = self.env.num_beams
            discrete_action_space_size = self.env.num_cells
            
            # 创建PPO智能体
            self.ppo_agent = PPO(
                state_dim=state_dim,
                discrete_action_dim=discrete_action_dim,
                continuous_action_dim=continuous_action_dim,
                discrete_action_space_size=discrete_action_space_size,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # 加载模型参数
            checkpoint = torch.load(model_path, map_location=self.ppo_agent.device)
            
            if 'discrete_actor' in checkpoint:
                # 新格式：字典包含各个组件的state_dict
                self.ppo_agent.discrete_actor.load_state_dict(checkpoint['discrete_actor'])
                self.ppo_agent.continuous_actor.load_state_dict(checkpoint['continuous_actor'])
                self.ppo_agent.critic.load_state_dict(checkpoint['critic'])
                print(f"成功加载模型: {model_path}")
            elif 'discrete_actor_state_dict' in checkpoint:
                # 旧格式：直接包含state_dict
                self.ppo_agent.discrete_actor.load_state_dict(checkpoint['discrete_actor_state_dict'])
                self.ppo_agent.continuous_actor.load_state_dict(checkpoint['continuous_actor_state_dict'])
                self.ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                print(f"加载模型成功 (兼容模式): {model_path}")
            else:
                # 最老格式：直接是state_dict（仅离散actor）
                self.ppo_agent.discrete_actor.load_state_dict(checkpoint)
                print(f"加载模型成功 (仅离散actor): {model_path}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.ppo_agent = None
    
    def evaluate_episode(self, algorithm: str, num_steps: int = 100) -> Dict:
        """
        评估单个episode
        
        Args:
            algorithm: 算法类型 ('ppo', 'greedy', 'random')
            num_steps: 评估步数
        
        Returns:
            episode_metrics: episode指标
        """
        # 重置环境
        state = self.env.reset()
        
        episode_rewards = []
        episode_throughput = []
        episode_delays = []
        episode_packet_losses = []
        episode_fairness = []
        episode_spectral_efficiency = []
        episode_power_efficiency = []
        
        for step in range(num_steps):
            # 根据算法类型选择动作
            if algorithm == 'ppo' and self.ppo_agent is not None:
                # PPO可能需要一维状态
                state_flat = state.flatten() if state.ndim > 1 else state
                action, _, _, _ = self.ppo_agent.select_action(state_flat)
            elif algorithm == 'greedy':
                traffic_demand = self.env.get_traffic_demand()
                action = self.baseline.greedy_allocation(traffic_demand)
            elif algorithm == 'random':
                action = self.baseline.random_allocation()
            else:
                raise ValueError(f"未知算法: {algorithm}")
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录指标 - 修正：直接从info中获取，而不是info['metrics']
            episode_rewards.append(reward)
            
            # 从info中直接获取指标（参考main.py的方式）
            if 'throughput' in info:
                episode_throughput.append(info['throughput'])
                episode_fairness.append(Metrics.calculate_fairness_index(info['data_rates']))
                episode_spectral_efficiency.append(Metrics.calculate_spectral_efficiency(info['data_rates'], 200.0))  # 默认带宽200MHz
                episode_power_efficiency.append(Metrics.calculate_power_efficiency(info['data_rates'], 39.0))  # 默认功率39dBW
            
            # 添加延迟和丢包指标
            if 'delay' in info:
                episode_delays.append(info['delay'])
            if 'packet_loss' in info:
                episode_packet_losses.append(info['packet_loss'])
            
            state = next_state
            
            if done:
                break
        
        # 计算episode统计
        episode_metrics = {
            'total_reward': np.sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards),
            'avg_throughput': np.mean(episode_throughput) if episode_throughput else 0,
            'avg_delays': np.mean(episode_delays) if episode_delays else 0,
            'avg_packet_losses': np.mean(episode_packet_losses) if episode_packet_losses else 0,
            'avg_fairness': np.mean(episode_fairness) if episode_fairness else 0,
            'avg_spectral_efficiency': np.mean(episode_spectral_efficiency) if episode_spectral_efficiency else 0,
            'avg_power_efficiency': np.mean(episode_power_efficiency) if episode_power_efficiency else 0,
            'num_steps': len(episode_rewards)
        }
        
        return episode_metrics
    
    def run_evaluation(self, num_episodes: int = 10, num_steps: int = 100):
        """
        运行完整评估
        
        Args:
            num_episodes: 评估episode数量
            num_steps: 每个episode的步数
        """
        algorithms = ['random', 'greedy']
        if self.ppo_agent is not None:
            algorithms.append('ppo')
        
        print(f"开始评估，将运行 {len(algorithms)} 种算法，每种算法 {num_episodes} 个episodes")
        
        for algorithm in algorithms:
            print(f"\n正在评估 {algorithm.upper()} 算法...")
            
            for episode in range(num_episodes):
                episode_metrics = self.evaluate_episode(algorithm, num_steps)
                  # 存储结果
                for key, value in episode_metrics.items():
                    self.results[algorithm][key].append(value)
                
                if(episode % 10 == 0):
                    print(f"Episode {episode + 1}/{num_episodes} - "
                        f"Reward: {episode_metrics['avg_reward']:.4f}, "
                        f"Throughput: {episode_metrics['avg_throughput']:.2f}, "
                        f"Delays: {episode_metrics['avg_delays']:.2f}, "
                        f"Packet Loss: {episode_metrics['avg_packet_losses']:.2f}, "
                        f"Spectral Eff: {episode_metrics['avg_spectral_efficiency']:.4f}")
    
    def print_summary(self):
        """打印评估结果摘要"""
        print("\n" + "="*80)
        print("评估结果摘要")
        print("="*80)
        
        algorithms = list(self.results.keys())
        metrics = ['avg_reward', 'avg_throughput', 'avg_delays', 'avg_packet_losses',
                  'avg_fairness', 'avg_spectral_efficiency', 'avg_power_efficiency']
        
        for metric in metrics:
            print(f"\n{metric}:")
            print("-" * 40)
            for algorithm in algorithms:
                if self.results[algorithm][metric]:
                    mean_val = np.mean(self.results[algorithm][metric])
                    std_val = np.std(self.results[algorithm][metric])
                    print(f"{algorithm.upper():>8}: {mean_val:8.4f} ± {std_val:8.4f}")
        
        # 打印相对性能
        if 'ppo' in self.results and self.results['ppo']['avg_reward']:
            print(f"\n相对性能提升 (相对于随机算法):")
            print("-" * 40)
            for algorithm in algorithms:
                if algorithm != 'random' and self.results[algorithm]['avg_reward']:
                    ppo_reward = np.mean(self.results[algorithm]['avg_reward'])
                    random_reward = np.mean(self.results['random']['avg_reward'])
                    improvement = (ppo_reward - random_reward) / abs(random_reward) * 100
                    print(f"{algorithm.upper():>8}: {improvement:+8.2f}%")
    
    def plot_results(self, save_path: str = "results"):
        """绘制评估结果"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
          # 设置颜色
        colors = {'random': 'red', 'greedy': 'blue', 'ppo': 'green'}
        
        algorithms = list(self.results.keys())
        metrics = ['avg_reward', 'avg_throughput', 'avg_delays', 'avg_packet_losses',
                  'avg_fairness', 'avg_spectral_efficiency', 'avg_power_efficiency']
          # 创建子图 - 调整为3x3以容纳7个指标
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 准备数据
            data = []
            labels = []
            
            for algorithm in algorithms:
                if self.results[algorithm][metric]:
                    data.append(self.results[algorithm][metric])
                    labels.append(algorithm.upper())
            
            # 绘制箱线图
            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                
                # 设置颜色
                for patch, label in zip(bp['boxes'], labels):
                    patch.set_facecolor(colors.get(label.lower(), 'gray'))
                    patch.set_alpha(0.7)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
          # 删除多余的子图
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'comparison_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 绘制性能对比柱状图
        self._plot_performance_comparison(save_path)
    
    def _plot_performance_comparison(self, save_path: str):
        """绘制性能对比柱状图"""
        algorithms = list(self.results.keys())
        metrics = ['avg_reward', 'avg_throughput', 'avg_spectral_efficiency', 'avg_power_efficiency']
        
        # 计算平均值
        means = {}
        stds = {}
        
        for algorithm in algorithms:
            means[algorithm] = []
            stds[algorithm] = []
            for metric in metrics:
                if self.results[algorithm][metric]:
                    means[algorithm].append(np.mean(self.results[algorithm][metric]))
                    stds[algorithm].append(np.std(self.results[algorithm][metric]))
                else:
                    means[algorithm].append(0)
                    stds[algorithm].append(0)
        
        # 绘制柱状图
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'random': 'red', 'greedy': 'blue', 'ppo': 'green'}
        
        for i, algorithm in enumerate(algorithms):
            offset = (i - len(algorithms)/2 + 0.5) * width
            bars = ax.bar(x + offset, means[algorithm], width, 
                         yerr=stds[algorithm], capsize=5,
                         label=algorithm.upper(), 
                         color=colors.get(algorithm, 'gray'),
                         alpha=0.8)
            
            # 添加数值标签
            for bar, mean_val in zip(bars, means[algorithm]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(stds[algorithm])*0.1,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('数值')
        ax.set_title('算法性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = "evaluation_results.txt"):
        """保存评估结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("模型评估结果\n")
            f.write("="*50 + "\n\n")
            
            algorithms = list(self.results.keys())
            metrics = ['avg_reward', 'avg_throughput', 'avg_delays', 'avg_packet_losses',
                      'avg_fairness', 'avg_spectral_efficiency', 'avg_power_efficiency']
            
            for metric in metrics:
                f.write(f"{metric}:\n")
                f.write("-" * 30 + "\n")
                for algorithm in algorithms:
                    if self.results[algorithm][metric]:
                        mean_val = np.mean(self.results[algorithm][metric])
                        std_val = np.std(self.results[algorithm][metric])
                        f.write(f"{algorithm.upper():>8}: {mean_val:8.4f} ± {std_val:8.4f}\n")
                f.write("\n")
            
            # 保存详细数据
            f.write("\n详细数据:\n")
            f.write("="*50 + "\n")
            for algorithm in algorithms:
                f.write(f"\n{algorithm.upper()}:\n")
                for metric in metrics:
                    if self.results[algorithm][metric]:
                        f.write(f"{metric}: {self.results[algorithm][metric]}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型评估和对比实验')
    parser.add_argument('--model_path', type=str, default='improved_models/ppo_model_final_improved.pt',
                      help='训练好的模型路径')
    parser.add_argument('--num_episodes', type=int, default=1000,
                      help='评估episode数量')
    parser.add_argument('--num_steps', type=int, default=100,                      help='每个episode的步数')
    parser.add_argument('--save_path', type=str, default='evaluation_results',
                      help='结果保存路径')
    parser.add_argument('--traffic_intensity', type=float, default=0.7,
                      help='评估时的流量强度')
    
    args = parser.parse_args()
    
    # 创建环境 - 使用与训练一致的参数
    print("创建环境...")
    env = SatelliteEnv(
        num_beams=4,
        num_cells=19,
        total_power_dbw=39.0,
        total_bandwidth_mhz=200.0,
        satellite_height_km=550.0,
        max_queue_size=150,  # 使用训练最后阶段的参数
        packet_ttl=16,      # 使用训练最后阶段的参数
        traffic_intensity=args.traffic_intensity  # 使用训练最后阶段的流量强度
    )
    
    # 创建评估器
    print("创建评估器...")
    evaluator = ModelEvaluator(env, args.model_path)
    
    # 运行评估
    evaluator.run_evaluation(args.num_episodes, args.num_steps)
    
    # 打印结果
    evaluator.print_summary()
    
    # 绘制结果
    evaluator.plot_results(args.save_path)
    
    # 保存结果
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # evaluator.save_results(os.path.join(args.save_path, 'evaluation_results.txt'))
    
    print(f"\n评估完成！结果已保存到 {args.save_path}")


if __name__ == "__main__":
    main()
