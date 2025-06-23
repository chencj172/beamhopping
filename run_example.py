import os
import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from environment.satellite_env import SatelliteEnv
from rl.ppo import PPO
from data.satellite_data import SatelliteData
from data.cell_data import CellData
from data.traffic_data import TrafficData
from utils.visualization import Visualization
from utils.metrics import Metrics

# 设置随机种子
np.random.seed(42)

# 创建输出目录
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 设置参数
num_beams = 4
num_cells = 19
satellite_height = 1000.0  # km
total_power = 39.0  # dBW
total_bandwidth = 200.0  # MHz
max_queue_size = 100
packet_ttl = 10
max_steps = 100

# 生成卫星位置数据
print("生成卫星位置数据...")
satellite_data = SatelliteData(
    satellite_height_km=satellite_height,
    inclination_deg=98.0,
    period_minutes=105.0,
    simulation_time_hours=24.0
)

# 生成小区位置数据
print("生成小区位置数据...")
cell_data = CellData(
    num_cells=num_cells,
    cell_radius_km=50.0,
    satellite_height_km=satellite_height
)

# 生成业务请求数据
print("生成业务请求数据...")
traffic_data = TrafficData(
    num_cells=num_cells,
    num_time_steps=1440,  # 24小时，每分钟一个时间步
    min_traffic_mbps=10.0,
    max_traffic_mbps=100.0,
    daily_pattern=True,
    random_seed=42
)

# 保存数据
satellite_data.save_to_file('results/satellite_positions.npy')
cell_data.save_to_file('results/cell_positions.npy')
traffic_data.save_to_file('results/traffic_data.npy')

# 可视化卫星轨道
print("可视化卫星轨道...")
satellite_data.plot_orbit()
plt.savefig('results/satellite_orbit.png')
plt.close()

# 可视化小区分布
print("可视化小区分布...")
cell_data.plot_cells_2d()
plt.savefig('results/cell_distribution_2d.png')
plt.close()

cell_data.plot_cells_3d(satellite_data.get_position(0))
plt.savefig('results/cell_distribution_3d.png')
plt.close()

# 可视化业务请求
print("可视化业务请求...")
traffic_data.plot_traffic_patterns([0, 5, 10, 15])
plt.savefig('results/traffic_patterns.png')
plt.close()

traffic_data.plot_traffic_heatmap(720)  # 中午12点的业务请求
plt.savefig('results/traffic_heatmap.png')
plt.close()

# 创建环境
print("创建环境...")
env = SatelliteEnv(
    num_beams=num_beams,
    num_cells=num_cells,
    total_power_dbw=total_power,
    total_bandwidth_mhz=total_bandwidth,
    satellite_height_km=satellite_height,
    max_queue_size=max_queue_size,
    packet_ttl=packet_ttl,
    reward_throughput_weight=0.7,
    reward_delay_weight=0.3
)

# 设置最大步数
env.max_steps = max_steps

# 获取状态和动作维度
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # 展平状态空间
discrete_action_dim = num_beams  # 离散动作维度（波束数量）
continuous_action_dim = num_beams  # 连续动作维度（波束数量）
discrete_action_space_size = num_cells  # 离散动作空间大小（小区数量）

# 创建PPO代理
print("创建PPO代理...")
agent = PPO(
    state_dim=state_dim,
    discrete_action_dim=discrete_action_dim,
    continuous_action_dim=continuous_action_dim,
    discrete_action_space_size=discrete_action_space_size,
    lr_actor=0.0003,
    lr_critic=0.001,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    update_epochs=10,
    mini_batch_size=64,
    device='cpu'
)

# 运行一个简短的训练示例
print("运行训练示例...")
num_episodes = 5

# 初始化指标
metrics = {
    'rewards': [],
    'throughputs': [],
    'delays': [],
    'packet_losses': [],
    'fairness': []
}

# 训练循环
for episode in range(num_episodes):
    # 重置环境
    state = env.reset()
    state = state.flatten()  # 展平状态
    
    # 初始化episode指标
    episode_reward = 0
    episode_throughput = 0
    episode_delay = 0
    episode_packet_loss = 0
    episode_fairness = 0
    
    # 运行一个episode
    for step in range(max_steps):
        # 选择动作
        action, action_log_prob, state_value = agent.select_action(state)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        next_state = next_state.flatten()  # 展平状态
        
        # 存储经验
        agent.buffer.add(
            state=state,
            discrete_action=action['beam_allocation'],
            continuous_action=action['power_allocation'],
            discrete_action_log_prob=action_log_prob,
            continuous_action_log_prob=action_log_prob,  # 简化处理，使用相同的对数概率
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
    
    # 存储指标
    metrics['rewards'].append(episode_reward)
    metrics['throughputs'].append(episode_throughput)
    metrics['delays'].append(episode_delay)
    metrics['packet_losses'].append(episode_packet_loss)
    metrics['fairness'].append(episode_fairness)
    
    # 打印训练信息
    print(f"Episode {episode+1}/{num_episodes} - "
          f"Reward: {episode_reward:.2f}, "
          f"Throughput: {episode_throughput:.2f} packets, "
          f"Delay: {episode_delay:.2f} slots, "
          f"Packet Loss: {episode_packet_loss:.2f} packets, "
          f"Fairness: {episode_fairness:.4f}")

# 保存模型
agent.save('models/ppo_model_example.pt')

# 可视化结果
print("可视化结果...")

# 绘制学习曲线
Visualization.plot_learning_curve(
    rewards=metrics['rewards'],
    avg_window=1,
    title='Learning Curve (Example)'
)
plt.savefig('results/learning_curve_example.png')
plt.close()

# 绘制性能指标
Visualization.plot_performance_metrics(
    metrics={
        'Throughput': metrics['throughputs'],
        'Delay': metrics['delays'],
        'Packet Loss': metrics['packet_losses'],
        'Fairness': metrics['fairness']
    },
    title='Performance Metrics (Example)'
)
plt.savefig('results/performance_metrics_example.png')
plt.close()

# 获取小区位置数据
cell_positions = cell_data.get_all_cell_positions()

# 将3D坐标投影到2D平面
cell_positions_2d = []
for pos in cell_positions:
    # 简单的投影方法
    x = pos[0]
    y = pos[1]
    cell_positions_2d.append([x, y])

cell_positions_2d = np.array(cell_positions_2d)

# 使用训练后的代理进行一次评估
print("评估训练后的代理...")
state = env.reset()
state = state.flatten()

# 获取代理的动作
action, _, _ = agent.select_action(state)

# 执行动作
_, _, _, info = env.step(action)

# 获取波束分配和功率分配
beam_allocation = action['beam_allocation']
power_allocation = action['power_allocation']

# 获取业务请求数据
traffic_demand = traffic_data.get_traffic_data(0)  # 使用第一个时间步的数据

# 绘制波束分配
Visualization.plot_beam_allocation(
    beam_allocation=beam_allocation,
    cell_positions=cell_positions_2d,
    power_allocation=power_allocation,
    traffic_demand=traffic_demand,
    title='Beam Allocation (Example)'
)
plt.savefig('results/beam_allocation_example.png')
plt.close()

# 绘制信噪比热力图
sinr = info['sinr']

Visualization.plot_sinr_heatmap(
    sinr=sinr,
    cell_positions=cell_positions_2d,
    beam_allocation=beam_allocation,
    title='SINR Heatmap (Example)'
)
plt.savefig('results/sinr_heatmap_example.png')
plt.close()

# 绘制队列状态
queue_lengths = info['queue_lengths']

Visualization.plot_queue_status(
    queue_lengths=queue_lengths,
    cell_positions=cell_positions_2d,
    max_queue_size=max_queue_size,
    title='Queue Status (Example)'
)
plt.savefig('results/queue_status_example.png')
plt.close()

print("示例运行完成！")
print("结果保存在 'results' 目录中，模型保存在 'models' 目录中。")