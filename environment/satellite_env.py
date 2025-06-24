import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, List, Optional

from environment.channel import ChannelModel
from environment.queue import QueueModel


class SatelliteEnv(gym.Env):
    """
    卫星跳波束环境类
    
    状态空间：卫星-小区信道增益矩阵和各小区的业务请求矩阵
    动作空间：波束分配矩阵（离散）和功率分配向量（连续）
    奖励：吞吐量和延迟的加权和
    """
    
    def __init__(self, 
                 num_beams: int = 4, 
                 num_cells: int = 19, 
                 total_power_dbw: float = 39.0,
                 total_bandwidth_mhz: float = 40.0,
                 satellite_height_km: float = 550.0,
                 max_queue_size: int = 100,
                 packet_ttl: int = 10,
                 reward_throughput_weight: float = 0.85,
                 reward_delay_weight: float = 0.15):
        """
        初始化卫星环境
        
        Args:
            num_beams: 波束数量
            num_cells: 小区数量
            total_power_dbw: 系统总功率(dBW)
            total_bandwidth_mhz: 系统总带宽(MHz)
            satellite_height_km: 卫星高度(km)
            max_queue_size: 队列最大容量
            packet_ttl: 数据包生存时间(时隙)
            reward_throughput_weight: 吞吐量奖励权重
            reward_delay_weight: 延迟奖励权重
        """
        super(SatelliteEnv, self).__init__()
        
        self.num_beams = num_beams
        self.num_cells = num_cells
        self.total_power_dbw = total_power_dbw
        self.total_power_linear = 10 ** (total_power_dbw / 10)  # 转换为线性值
        self.total_bandwidth_mhz = total_bandwidth_mhz
        self.satellite_height_km = satellite_height_km
        self.max_queue_size = max_queue_size
        self.packet_ttl = packet_ttl
        self.reward_throughput_weight = reward_throughput_weight
        self.reward_delay_weight = reward_delay_weight
        
        # 创建信道模型
        self.channel_model = ChannelModel(
            num_beams=num_beams,
            num_cells=num_cells,
            satellite_height_km=satellite_height_km
        )
        
        # 创建队列模型
        self.queue_model = QueueModel(
            num_cells=num_cells,
            max_queue_size=max_queue_size,
            packet_ttl=packet_ttl
        )
        
        # 定义观测空间
        # 观测包括：信道增益矩阵 (num_beams x num_cells) 和业务请求矩阵 (num_cells)
        obs_low = np.zeros((num_beams + 1, num_cells))
        obs_high = np.ones((num_beams + 1, num_cells)) * float('inf')
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 定义动作空间
        # 动作包括：波束分配矩阵（离散）和功率分配向量（连续）
        # 波束分配：每个波束可以分配给哪个小区，共 num_beams 个波束
        self.beam_allocation_space = spaces.MultiDiscrete([num_cells] * num_beams)
        
        # 功率分配：每个波束分配多少功率，总和不超过总功率
        self.power_allocation_space = spaces.Box(
            low=np.zeros(num_beams),
            high=np.ones(num_beams),  # 归一化的功率分配比例
            dtype=np.float32
        )
        
        # 组合动作空间
        self.action_space = spaces.Dict({
            'beam_allocation': self.beam_allocation_space,
            'power_allocation': self.power_allocation_space
        })
        
        # 初始化状态
        self.channel_gains = None  # 信道增益矩阵
        self.traffic_requests = None  # 业务请求矩阵
        self.current_step = 0
        self.max_steps = 1000  # 每个episode的最大步数
        
        # 性能指标
        self.throughput_history = []
        self.delay_history = []
        self.packet_loss_history = []
    
    def reset(self):
        """
        重置环境状态
        
        Returns:
            observation: 初始观测
        """
        # 重置当前步数
        self.current_step = 0
        
        # 重置信道模型
        self.channel_model.reset()
        self.channel_gains = self.channel_model.get_channel_gains()
        
        # 重置队列模型
        self.queue_model.reset()
        self.traffic_requests = self.queue_model.get_queue_state()
        
        # 重置性能指标
        self.throughput_history = []
        self.delay_history = []
        self.packet_loss_history = []
        
        # 返回初始观测
        return self._get_observation()
    
    def step(self, action: Dict):
        """
        执行一步环境交互
        
        Args:
            action: 动作字典，包含波束分配和功率分配
                - beam_allocation: 波束分配数组，shape=(num_beams,)
                - power_allocation: 功率分配数组，shape=(num_beams,)
        
        Returns:
            observation: 新的观测
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 解析动作
        beam_allocation = action['beam_allocation']  # 波束分配，整数数组
        power_allocation = action['power_allocation']  # 功率分配，浮点数数组
        
        # 确保beam_allocation和power_allocation是一维数组
        beam_allocation = np.array(beam_allocation).flatten()
        power_allocation = np.array(power_allocation).flatten()
        
        # 确保power_allocation的长度正确
        if len(power_allocation) != self.num_beams:
            # 如果长度不正确，使用均匀分配
            power_allocation = np.ones(self.num_beams) / self.num_beams
        
        # 确保功率值非负
        power_allocation = np.maximum(power_allocation, 0.0)
        
        # 归一化功率分配
        power_sum = np.sum(power_allocation)
        power_allocation = power_allocation / power_sum if power_sum > 0 else np.ones(self.num_beams) / self.num_beams

        # 计算实际功率分配（线性值）
        actual_power = power_allocation * self.total_power_linear

        # 计算信噪比和数据率
        sinr, data_rates = self._calculate_sinr_and_data_rates(beam_allocation, actual_power)

        # 更新队列状态
        served_packets, dropped_packets, avg_delay = self.queue_model.update(data_rates)
        
        # 生成新的业务请求
        self.queue_model.generate_traffic()
        
        # 更新信道状态
        self.channel_model.update()
        self.channel_gains = self.channel_model.get_channel_gains()
        
        # 获取新的业务请求状态
        self.traffic_requests = self.queue_model.get_queue_state()

        # print(self.traffic_requests)
        
        # 计算奖励
        reward = self._calculate_reward(served_packets, avg_delay, dropped_packets)
        
        # 更新性能指标
        throughput = np.sum(served_packets)  # 总吞吐量（服务的数据包数量）
        self.throughput_history.append(throughput)
        self.delay_history.append(avg_delay)
        self.packet_loss_history.append(dropped_packets)
        
        # 更新当前步数
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 返回观测、奖励、是否结束和额外信息
        return self._get_observation(), reward, done, {
            'throughput': throughput,
            'delay': avg_delay,
            'packet_loss': dropped_packets,
            'sinr': sinr,
            'data_rates': data_rates
        }
    
    def _get_observation(self):
        """
        获取当前观测
        
        Returns:
            observation: 当前观测，包含信道增益矩阵和业务请求矩阵
        """
        # 将信道增益矩阵和业务请求矩阵组合成观测
        observation = np.vstack([
            self.channel_gains,  # 信道增益矩阵 (num_beams x num_cells)
            self.traffic_requests.reshape(1, -1)  # 业务请求矩阵 (1 x num_cells)
        ])
        
        return observation
    
    def _calculate_sinr_and_data_rates(self, beam_allocation, power_allocation):
        """
        计算信噪比和数据率
        
        Args:
            beam_allocation: 波束分配数组
            power_allocation: 功率分配数组
        
        Returns:
            sinr: 信噪比数组
            data_rates: 数据率数组
        """
        # 初始化信噪比和数据率数组
        sinr = np.zeros(self.num_cells)
        data_rates = np.zeros(self.num_cells)
        
        # 确保beam_allocation是一维数组
        beam_allocation = np.array(beam_allocation).flatten()
        
        # 噪声功率密度 (dBW/Hz)
        noise_power_density_dbw_hz = -174 - 30  # -174 dBm/Hz 转换为 dBW/Hz（-30dB是mW到W的转换）
        
        # 每个波束的带宽 (Hz)
        bandwidth_hz = self.total_bandwidth_mhz * 1e6
        
        # 每个小区的噪声功率 (线性值)
        noise_power = 10 ** (noise_power_density_dbw_hz / 10) * bandwidth_hz

        BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant in J/K
        T_rx = 290  # 接收机温度 (K)
        thermal_noise = BOLTZMANN_CONSTANT * T_rx * bandwidth_hz  # 热噪声 (W)
        
        # 创建波束到小区的映射
        beam_to_cell = {}
        for beam_idx, cell_idx in enumerate(beam_allocation):
            # 确保cell_idx是有效的小区索引
            if 0 <= cell_idx < self.num_cells:
                beam_to_cell[beam_idx] = int(cell_idx)
        
        # 计算每个小区的信号功率和干扰功率
        for cell_idx in range(self.num_cells):
            # 初始化信号功率和干扰功率
            signal_power = 0
            interference_power = 0
            
            # 检查该小区是否被分配了波束
            serving_beam = None
            for beam_idx, served_cell in beam_to_cell.items():
                if served_cell == cell_idx:
                    serving_beam = beam_idx
                    break
            
            # 如果该小区被分配了波束
            if serving_beam is not None:
                # 计算信号功率（确保是标量值）
                power = float(power_allocation[serving_beam])
                gain = float(self.channel_gains[serving_beam, cell_idx])
                signal_power = power * gain
                
                # 计算来自其他波束的干扰
                for beam_idx, beam_power in enumerate(power_allocation):
                    if beam_idx != serving_beam:  # 排除服务波束
                        beam_power_float = float(beam_power)
                        beam_gain_float = float(self.channel_gains[beam_idx, cell_idx])
                        interference_power += beam_power_float * beam_gain_float
                
                # 计算信噪比（确保是标量值）
                sinr[cell_idx] = signal_power / (interference_power + thermal_noise)
                # 只在前5个时间步打印调试信息，避免过多输出
                # print(f"Cell {cell_idx}: SINR={sinr[cell_idx]:.4f}, Signal={signal_power:.4e}, Interference={interference_power:.4e}, Noise={noise_power:.4e}")
                # 计算数据率 (Shannon公式，bits/s)
                data_rates[cell_idx] = bandwidth_hz * np.log2(1 + sinr[cell_idx])
        return sinr, data_rates
        
    # 为了可视化，允许外部直接调用计算信噪比的方法
    def _calculate_sinr(self, beam_allocation, power_allocation):
        """
        计算信噪比（用于可视化）
        
        Args:
            beam_allocation: 波束分配数组
            power_allocation: 功率分配数组
            
        Returns:
            sinr: 信噪比数组
            data_rates: 数据率数组
        """
        return self._calculate_sinr_and_data_rates(beam_allocation, power_allocation)
    
    def _calculate_reward(self, served_packets, avg_delay, dropped_packets):
        """
        计算奖励
        
        Args:
            served_packets: 服务的数据包数量
            avg_delay: 平均延迟
            dropped_packets: 丢弃的数据包数量
        
        Returns:
            reward: 奖励值
        """
        # 计算吞吐量奖励（归一化）
        throughput_reward = np.sum(served_packets)
        # throughput_reward = np.sum(served_packets) / (self.num_cells * self.max_queue_size)
        # 计算延迟惩罚（归一化）
        delay_penalty = avg_delay / self.packet_ttl if avg_delay > 0 else 0
        
        # 计算丢包惩罚（归一化）
        # drop_penalty = dropped_packets / (self.num_cells * self.max_queue_size)
        drop_penalty = dropped_packets
        
        # 计算总奖励
        # reward = (self.reward_throughput_weight * throughput_reward) - \
        #          (self.reward_delay_weight * (delay_penalty + drop_penalty))
        
        reward = (throughput_reward * 0.8) 
        - \
                 (0.2 * delay_penalty + 2.0 * drop_penalty)
        
        return reward
    
    def render(self, mode='human'):
        """
        渲染环境
        
        Args:
            mode: 渲染模式
        """
        pass  # 可以在这里实现可视化功能
    
    def close(self):
        """
        关闭环境
        """
        pass
    
    def get_performance_metrics(self):
        """
        获取性能指标
        
        Returns:
            metrics: 性能指标字典
        """
        return {
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'avg_delay': np.mean(self.delay_history) if self.delay_history else 0,
            'avg_packet_loss': np.mean(self.packet_loss_history) if self.packet_loss_history else 0
        }