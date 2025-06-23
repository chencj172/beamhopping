import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time

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
    解析命令行参数
    
    Returns:
        args: 参数对象
    """
    parser = argparse.ArgumentParser(description='Satellite Beam Hopping Experiment')
    
    # 实验设置
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=1200, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save interval')
    
    # 环境参数
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams')
    parser.add_argument('--num_cells', type=int, default=19, help='Number of cells')
    parser.add_argument('--satellite_height', type=float, default=550.0, help='Satellite height (km)')
    parser.add_argument('--total_power', type=float, default=39.0, help='Total power (dBW)')
    parser.add_argument('--total_bandwidth', type=float, default=200.0, help='Total bandwidth (MHz)')
    
    # 队列参数
    parser.add_argument('--max_queue_size', type=int, default=300, help='Maximum queue size')
    parser.add_argument('--packet_ttl', type=int, default=20, help='Packet time-to-live (slots)')
    parser.add_argument('--packet_size', type=float, default=100.0, help='Packet size (KB)')
    
    # 强化学习参数
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.97, help='GAE lambda parameter')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of update epochs')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini-batch size')
    
    # 奖励权重
    parser.add_argument('--reward_throughput_weight', type=float, default=0.70, help='Throughput reward weight')
    parser.add_argument('--reward_delay_weight', type=float, default=0.30, help='Delay reward weight')
    
    # 数据参数
    parser.add_argument('--generate_data',default=True, action='store_true', help='是否重新生成数据，默认从文件加载')
    
    # 输出设置
    parser.add_argument('--data_dir', type=str, default='data_dir', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Model directory')
    parser.add_argument('--verbose',default=True, action='store_true', help='Verbose output')
    
    return parser.parse_args()


def setup_environment(args):
    """
    设置实验环境
    
    Args:
        args: 参数对象
    
    Returns:
        env: 环境对象
    """
    # 创建环境
    env = SatelliteEnv(
        num_beams=args.num_beams,
        num_cells=args.num_cells,
        total_power_dbw=args.total_power,
        total_bandwidth_mhz=args.total_bandwidth,
        satellite_height_km=args.satellite_height,
        max_queue_size=args.max_queue_size,
        packet_ttl=args.packet_ttl,
        reward_throughput_weight=args.reward_throughput_weight,
        reward_delay_weight=args.reward_delay_weight
    )
    
    # 设置最大步数
    env.max_steps = args.max_steps
    
    return env


def setup_agent(env, args):
    """
    设置强化学习代理
    
    Args:
        env: 环境对象
        args: 参数对象
    
    Returns:
        agent: 代理对象
    """
    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # 展平状态空间
    discrete_action_dim = args.num_beams  # 离散动作维度（波束数量）
    continuous_action_dim = args.num_beams  # 连续动作维度（波束数量）
    discrete_action_space_size = args.num_cells  # 离散动作空间大小（小区数量）
    
    # print(discrete_action_dim, "==", continuous_action_dim, "==", discrete_action_space_size)
    # 创建PPO代理
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
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return agent


def generate_data(args, load_from_file=True):
    """
    生成或加载实验数据
    
    Args:
        args: 参数对象
        load_from_file: 是否从文件加载数据，默认为True
    
    Returns:
        data: 数据字典
    """
    # 创建输出目录
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 数据文件路径
    # .npy文件路径（用于加载数据）
    satellite_npy_path = os.path.join(args.data_dir, 'satellite_positions_fixed.npy')
    cell_npy_path = os.path.join(args.data_dir, 'cell_positions_fixed.npy')
    traffic_npy_path = os.path.join(args.data_dir, 'traffic_data_custom.npy')
    
    # CSV文件路径（用于保存数据）
    satellite_csv_path = os.path.join(args.data_dir, 'satellite_positions_fixed.csv')
    cell_csv_path = os.path.join(args.data_dir, 'cell_positions_fixed.csv')
    traffic_csv_path = os.path.join(args.data_dir, 'traffic_data_custom.csv')
    
    # 检查是否存在.npy文件，优先从.npy文件加载
    if load_from_file and os.path.exists(satellite_npy_path) and os.path.exists(cell_npy_path) and os.path.exists(traffic_npy_path):
        print("从.npy文件加载数据...")
        # 从.npy文件加载数据
        satellite_data = SatelliteDataFixed.load_from_file(satellite_npy_path)
        cell_data = CellDataFixed.load_from_file(cell_npy_path)
        traffic_data = TrafficDataPoisson.load_from_file(traffic_npy_path)
        print("数据加载完成！")
    # 如果.npy文件不存在，但CSV文件存在，则从CSV文件加载
    elif load_from_file and os.path.exists(satellite_csv_path) and os.path.exists(cell_csv_path) and os.path.exists(traffic_csv_path):
        print("从CSV文件加载数据...")
        # 由于SatelliteDataFixed和CellDataFixed没有load_from_csv方法，我们先创建对象
        satellite_data = SatelliteDataFixed(satellite_height_km=args.satellite_height)
        cell_data = CellDataFixed(num_cells=args.num_cells, cell_radius_km=50.0, satellite_height_km=args.satellite_height)
        
        # 加载业务请求数据
        traffic_data = TrafficDataPoisson.load_from_csv(traffic_csv_path)
        print("数据加载完成！")
    else:
        print("生成新数据...")
        # 生成固定卫星位置数据
        satellite_data = SatelliteDataFixed(
            satellite_height_km=args.satellite_height
        )
        
        # 生成小区位置数据
        cell_data = CellDataFixed(
            num_cells=args.num_cells,
            cell_radius_km=50.0,
            satellite_height_km=args.satellite_height
        )
        
        # 生成业务请求数据（自定义泊松分布）
        traffic_data = TrafficDataPoisson(
            num_cells=args.num_cells,
            num_time_steps=86400,  # 24小时，每秒一个时间步
            daily_pattern=True,
            random_seed=args.seed
        )
        
        # 保存为.npy格式（用于快速加载）
        satellite_data.save_to_file(satellite_npy_path)
        cell_data.save_to_file(cell_npy_path)
        traffic_data.save_to_file(traffic_npy_path)
        
        # 保存为CSV格式（用于查看和分析）
        satellite_data.save_to_csv(satellite_csv_path)
        cell_data.save_to_csv(cell_csv_path)
        traffic_data.save_to_csv(traffic_csv_path)
        
        print("数据生成并保存完成！")
    
    # 返回数据字典
    data = {
        'satellite_data': satellite_data,
        'cell_data': cell_data,
        'traffic_data': traffic_data
    }
    
    return data


def train(env, agent, args):
    """
    训练强化学习代理
    
    Args:
        env: 环境对象
        agent: 代理对象
        args: 参数对象
    
    Returns:
        metrics: 训练指标字典
    """
    # 创建模型目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 初始化指标
    metrics = {
        'rewards': [],
        'throughputs': [],
        'delays': [],
        'packet_losses': [],
        'fairness': []
    }
    
    # 训练循环
    for episode in tqdm(range(args.num_episodes), desc='Training'):
        # 重置环境
        state = env.reset()
        state = state.flatten()  # 展平状态
        
        # 初始化episode指标
        episode_reward = 0
        episode_throughput = 0
        episode_delay = 0
        episode_packet_loss = 0
        episode_fairness = 0
        episode_power = 0

        # 运行一个episode
        for step in range(args.max_steps):
            # 选择动作
            action, discrete_action_log_prob, continuous_action_log_prob, state_value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            next_state = next_state.flatten()  # 展平状态
            
            # 存储经验
            agent.buffer.add(
                state=state,
                discrete_action=action['beam_allocation'],
                continuous_action=action['power_allocation'],
                discrete_action_log_prob=discrete_action_log_prob,
                continuous_action_log_prob=continuous_action_log_prob,
                reward=reward,
                value=state_value,
                done=done
            )
            
            # 更新状态
            state = next_state
            
            # 更新episode指标
            episode_reward += reward
            episode_throughput += info['throughput']
            episode_delay += info['delay']
            episode_packet_loss += info['packet_loss']
            episode_fairness += Metrics.calculate_fairness_index(info['data_rates'])
            episode_power += Metrics.calculate_power_efficiency(info['data_rates'], args.total_power)
            
            # 如果episode结束，跳出循环
            if done:
                break
        
        # 更新策略
        loss_info = agent.update()
        
        # 计算平均指标
        episode_throughput /= (step + 1)
        episode_delay /= (step + 1)
        episode_packet_loss /= (step + 1)
        episode_fairness /= (step + 1)
        episode_power /= (step + 1)
        
        # 存储指标
        metrics['rewards'].append(episode_reward)
        metrics['throughputs'].append(episode_throughput)
        metrics['delays'].append(episode_delay)
        metrics['packet_losses'].append(episode_packet_loss)
        metrics['fairness'].append(episode_fairness)
        metrics['power_efficiency'] = episode_power
        
        # 打印训练信息
        if args.verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{args.num_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Throughput: {episode_throughput:.2f} packets, "
                  f"Delay: {episode_delay:.2f} slots, "
                  f"Packet Loss: {episode_packet_loss:.2f} packets, ")
        
        # 评估代理
        if (episode + 1) % args.eval_interval == 0:
            evaluate(env, agent, episode + 1, args)
        
        # 保存模型
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(args.model_dir, f'ppo_model_episode_{episode+1}.pt'))
    
    # 保存最终模型
    agent.save(os.path.join(args.model_dir, 'ppo_model_final.pt'))
    
    return metrics


def evaluate(env, agent, episode, args):
    """
    评估强化学习代理
    
    Args:
        env: 环境对象
        agent: 代理对象
        episode: 当前episode
        args: 参数对象
    
    Returns:
        eval_metrics: 评估指标字典
    """
    # 初始化评估指标
    eval_metrics = {
        'reward': 0,
        'throughput': 0,
        'delay': 0,
        'packet_loss': 0,
        'fairness': 0,
        'spectral_efficiency': 0,
        'power_efficiency': 0
    }
    
    # 重置环境
    state = env.reset()
    state = state.flatten()  # 展平状态
    
    # 运行一个episode
    for step in range(args.max_steps):
        # 选择动作（评估模式，不存储经验）
        with torch.no_grad():
            action, _, _, _ = agent.select_action(state)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        next_state = next_state.flatten()  # 展平状态
        
        # 更新状态
        state = next_state
        
        # 更新评估指标
        eval_metrics['reward'] += reward
        eval_metrics['throughput'] += info['throughput']
        eval_metrics['delay'] += info['delay']
        eval_metrics['packet_loss'] += info['packet_loss']
        eval_metrics['fairness'] += Metrics.calculate_fairness_index(info['data_rates'])
        eval_metrics['spectral_efficiency'] += Metrics.calculate_spectral_efficiency(info['data_rates'], args.total_bandwidth)
        eval_metrics['power_efficiency'] += Metrics.calculate_power_efficiency(info['data_rates'], args.total_power)
        
        # 如果episode结束，跳出循环
        if done:
            break
    
    # 计算平均指标
    for key in eval_metrics:
        eval_metrics[key] /= (step + 1)
    
    # 打印评估信息
    if args.verbose:
        print(f"\nEvaluation at Episode {episode}/{args.num_episodes}:")
        print(f"Reward: {eval_metrics['reward']:.2f}")
        print(f"Throughput: {eval_metrics['throughput']:.2f} packets")
        print(f"Delay: {eval_metrics['delay']:.2f} slots")
        print(f"Packet Loss: {eval_metrics['packet_loss']:.2f} packets")
        # print(f"Fairness: {eval_metrics['fairness']:.4f}")
        print(f"Spectral Efficiency: {eval_metrics['spectral_efficiency']:.4f} bits/s/Hz")
        print(f"Power Efficiency: {eval_metrics['power_efficiency']:.4f} bits/s/W\n")
    
    return eval_metrics


def visualize_results(metrics, data, args):
    """
    可视化实验结果
    
    Args:
        metrics: 训练指标字典
        data: 数据字典
        args: 参数对象
    """
    # 绘制学习曲线
    Visualization.plot_learning_curve(
        rewards=metrics['rewards'],
        avg_window=10,
        title='Learning Curve'
    )
    plt.savefig(os.path.join(args.output_dir, 'learning_curve_fixed.png'))
    plt.close()
    
    # 绘制性能指标
    Visualization.plot_performance_metrics(
        metrics={
            'Throughput': metrics['throughputs'],
            'Delay': metrics['delays'],
            'Packet Loss': metrics['packet_losses'],
            'Fairness': metrics['fairness']
        },
        title='Performance Metrics'
    )
    plt.savefig(os.path.join(args.output_dir, 'performance_metrics_fixed.png'))
    plt.close()
    
    # 获取小区位置数据
    cell_positions = data['cell_data'].get_all_cell_positions()
    
    # 将3D坐标投影到2D平面
    cell_positions_2d = []
    for pos in cell_positions:
        # 简单的投影方法
        x = pos[0]
        y = pos[1]
        cell_positions_2d.append([x, y])
    
    cell_positions_2d = np.array(cell_positions_2d)
    
    # 绘制波束分配示例
    # 这里使用随机生成的波束分配和功率分配作为示例
    beam_allocation = np.random.randint(0, args.num_cells, size=args.num_beams)
    power_allocation = np.random.uniform(0, 1, size=args.num_beams)
    power_allocation = power_allocation / np.sum(power_allocation)  # 归一化
    
    # 获取业务请求数据
    traffic_demand = data['traffic_data'].get_traffic_data(3600)  # 使用1小时时间点的数据（秒级时间步）
    
    # 绘制波束分配
    Visualization.plot_beam_allocation(
        beam_allocation=beam_allocation,
        cell_positions=cell_positions_2d,
        power_allocation=power_allocation,
        traffic_demand=traffic_demand,
        title='Beam Allocation'
    )
    plt.savefig(os.path.join(args.output_dir, 'beam_allocation_fixed.png'))
    plt.close()
    
    # 绘制信噪比热力图
    # 这里使用随机生成的信噪比作为示例
    sinr = np.random.uniform(-10, 30, size=args.num_cells)  # dB
    
    Visualization.plot_sinr_heatmap(
        sinr=sinr,
        cell_positions=cell_positions_2d,
        beam_allocation=beam_allocation,
        title='SINR Heatmap (Fixed Satellite Position)'
    )
    plt.savefig(os.path.join(args.output_dir, 'sinr_heatmap_fixed.png'))
    plt.close()
    
    # 绘制队列状态
    # 这里使用随机生成的队列长度作为示例
    queue_lengths = np.random.randint(0, args.max_queue_size, size=args.num_cells)
    
    Visualization.plot_queue_status(
        queue_lengths=queue_lengths,
        cell_positions=cell_positions_2d,
        max_queue_size=args.max_queue_size,
        title='Queue Status (Fixed Satellite Position)'
    )
    plt.savefig(os.path.join(args.output_dir, 'queue_status_fixed.png'))
    plt.close()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 生成或加载实验数据
    print("Generating or loading experimental data...")
    data = generate_data(args, args.generate_data)  # 如果generate_data为True，则不从文件加载

    # 设置环境
    # print("Setting up environment...")
    env = setup_environment(args)
    
    # 设置代理
    # print("Setting up agent...")
    agent = setup_agent(env, args)
    
    # 训练代理
    print("Training agent...")
    start_time = time.time()
    metrics = train(env, agent, args)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # 可视化结果
    print("Visualizing results...")
    visualize_results(metrics, data, args)
    
    print("Experiment completed.")


if __name__ == "__main__":
    main()