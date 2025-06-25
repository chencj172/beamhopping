import numpy as np
import gym
from gym import spaces
from typing import Dict, Tuple, List, Optional

from environment.channel import ChannelModel
from environment.queue import QueueModel


class DynamicChannelModel(ChannelModel):
    """
    动态信道模型类 - 继承自基础信道模型，增加动态特性
    """
    def __init__(self, *args, **kwargs):
        # 添加动态信道参数
        self.fading_enabled = kwargs.pop('fading_enabled', True)
        self.shadowing_enabled = kwargs.pop('shadowing_enabled', True)
        self.weather_enabled = kwargs.pop('weather_enabled', True)
        self.satellite_movement_enabled = kwargs.pop('satellite_movement_enabled', True)
        
        # 信道衰落参数
        self.rayleigh_scale = kwargs.pop('rayleigh_scale', 0.5)  # Rayleigh衰落尺度参数
        self.rice_k_factor = kwargs.pop('rice_k_factor', 10.0)   # Rice衰落K因子
        self.shadowing_std = kwargs.pop('shadowing_std', 4.0)    # 阴影衰落标准差(dB)
        
        # 天气衰减参数
        self.rain_rate = kwargs.pop('rain_rate', 0.0)  # 降雨率 (mm/h)
        self.cloud_attenuation = kwargs.pop('cloud_attenuation', 0.0)  # 云层衰减(dB)
        
        # 卫星轨道参数
        self.orbital_period = kwargs.pop('orbital_period', 5760)  # 轨道周期(时隙)
        self.orbital_inclination = kwargs.pop('orbital_inclination', 97.4)  # 轨道倾角(度)
        
        # 初始化动态状态（在调用父类之前）
        self.current_time = 0
        self.fading_state = None
        self.shadowing_state = None
        self.weather_attenuation = None
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 现在设置动态状态的实际值
        self.fading_state = np.ones((self.num_beams, self.num_cells))
        self.shadowing_state = np.zeros((self.num_beams, self.num_cells))
        self.weather_attenuation = np.zeros(self.num_cells)
        
        # 卫星初始位置和速度
        self.satellite_velocity = self._calculate_orbital_velocity()
        self.initial_position = self.satellite_position.copy()
    
    def _calculate_orbital_velocity(self):
        """计算卫星轨道速度"""
        # 地球引力参数 (m^3/s^2)
        mu = 3.986004418e14
        # 卫星轨道半径 (m)
        r = np.linalg.norm(self.satellite_position)
        # 轨道速度 (m/s)
        v = np.sqrt(mu / r)
        # 速度方向（简化为垂直于位置向量）
        pos_norm = self.satellite_position / np.linalg.norm(self.satellite_position)
        # 在轨道平面内垂直方向
        velocity_direction = np.array([-pos_norm[1], pos_norm[0], 0])
        velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
        
        return velocity_direction * v
    
    def update(self):
        """更新动态信道状态"""
        self.current_time += 1
        
        # 更新卫星位置
        if self.satellite_movement_enabled:
            self._update_satellite_position()
        
        # 更新信道衰落
        if self.fading_enabled:
            self._update_fading()
        
        # 更新阴影衰落
        if self.shadowing_enabled:
            self._update_shadowing()
        
        # 更新天气条件
        if self.weather_enabled:
            self._update_weather()
        
        # 重新计算信道增益
        self._update_channel_gains()
    
    def _update_satellite_position(self):
        """更新卫星位置"""
        # 时隙长度（秒）
        time_slot_seconds = 1.0
        
        # 更新位置
        self.satellite_position += self.satellite_velocity * time_slot_seconds
        
        # 简化的轨道模型：周期性运动
        orbit_angle = 2 * np.pi * self.current_time / self.orbital_period
        orbit_radius = np.linalg.norm(self.initial_position)
        
        # 新的轨道位置（简化为圆形轨道）
        self.satellite_position[0] = orbit_radius * np.cos(orbit_angle)
        self.satellite_position[1] = orbit_radius * np.sin(orbit_angle)
        # Z坐标保持相对稳定，但有小幅变化
        self.satellite_position[2] = self.initial_position[2] * (1 + 0.1 * np.sin(orbit_angle * 3))
    
    def _update_fading(self):
        """更新多径衰落"""
        for beam_idx in range(self.num_beams):
            for cell_idx in range(self.num_cells):
                # 使用Rice衰落模型（包含视距分量）
                # Rice分布：有视距分量的衰落模型
                if self.rice_k_factor > 0:
                    # Rice衰落
                    los_component = np.sqrt(self.rice_k_factor / (self.rice_k_factor + 1))
                    nlos_component = np.sqrt(1 / (self.rice_k_factor + 1))
                    
                    # 生成复高斯随机变量
                    real_part = np.random.normal(los_component, nlos_component * self.rayleigh_scale)
                    imag_part = np.random.normal(0, nlos_component * self.rayleigh_scale)
                    
                    # 计算幅度（衰落系数）
                    fading_amplitude = np.sqrt(real_part**2 + imag_part**2)
                else:
                    # Rayleigh衰落（无视距分量）
                    fading_amplitude = np.random.rayleigh(self.rayleigh_scale)
                
                # 更新衰落状态（应用时间相关性）
                correlation = 0.9  # 时间相关性系数
                self.fading_state[beam_idx, cell_idx] = (
                    correlation * self.fading_state[beam_idx, cell_idx] +
                    (1 - correlation) * fading_amplitude
                )
    
    def _update_shadowing(self):
        """更新阴影衰落"""
        # 阴影衰落通常变化较慢，具有空间相关性
        for beam_idx in range(self.num_beams):
            for cell_idx in range(self.num_cells):
                # 生成相关的对数正态分布阴影衰落
                correlation = 0.95  # 时间相关性系数
                innovation = np.random.normal(0, self.shadowing_std * np.sqrt(1 - correlation**2))
                
                self.shadowing_state[beam_idx, cell_idx] = (
                    correlation * self.shadowing_state[beam_idx, cell_idx] + innovation
                )
    
    def _update_weather(self):
        """更新天气条件"""
        # 动态天气模型
        for cell_idx in range(self.num_cells):
            # 降雨衰减（ITU-R P.618模型简化版）
            if self.rain_rate > 0:
                # 频率相关的降雨衰减系数
                if self.frequency_ghz < 10:
                    rain_coeff = 0.0001 * self.frequency_ghz**2.42
                else:
                    rain_coeff = 0.003 * self.frequency_ghz**1.75
                
                rain_attenuation = rain_coeff * self.rain_rate  # dB
            else:
                rain_attenuation = 0
            
            # 云层衰减
            cloud_attenuation = self.cloud_attenuation
            
            # 总天气衰减
            total_weather_attenuation = rain_attenuation + cloud_attenuation
            
            # 添加随机变化
            weather_variation = np.random.normal(0, 0.5)  # 0.5dB标准差
            self.weather_attenuation[cell_idx] = total_weather_attenuation + weather_variation
            
            # 确保衰减为正值
            self.weather_attenuation[cell_idx] = max(0, self.weather_attenuation[cell_idx])
    def _update_channel_gains(self):
        """更新信道增益矩阵（考虑所有动态效应）"""
        # 首先计算基础信道增益
        super()._update_channel_gains()
        
        # 如果动态状态还未初始化，直接返回
        if (self.fading_state is None or 
            self.shadowing_state is None or 
            self.weather_attenuation is None):
            return
        
        # 应用动态效应
        for beam_idx in range(self.num_beams):
            for cell_idx in range(self.num_cells):
                # 基础增益（线性值）
                base_gain = self.channel_gains[beam_idx, cell_idx]
                
                # 应用多径衰落
                fading_factor = self.fading_state[beam_idx, cell_idx]
                
                # 应用阴影衰落（转换为线性值）
                shadowing_factor = 10 ** (self.shadowing_state[beam_idx, cell_idx] / 10)
                
                # 应用天气衰减（转换为线性值）
                weather_factor = 10 ** (-self.weather_attenuation[cell_idx] / 10)
                
                # 计算最终信道增益
                self.channel_gains[beam_idx, cell_idx] = (
                    base_gain * fading_factor * shadowing_factor * weather_factor
                )
    def set_weather_conditions(self, rain_rate: float, cloud_attenuation: float):
        """设置天气条件"""
        self.rain_rate = rain_rate
        self.cloud_attenuation = cloud_attenuation
    
    def get_channel_state_info(self):
        """获取详细的信道状态信息"""
        return {
            'fading_state': self.fading_state.copy(),
            'shadowing_state': self.shadowing_state.copy(),
            'weather_attenuation': self.weather_attenuation.copy(),
            'satellite_position': self.satellite_position.copy(),
            'current_time': self.current_time
        }
    
    def get_current_channel_state(self):
        """获取当前信道状态（信道增益的摘要）"""
        # 返回每个小区的平均信道增益
        if self.channel_gains is not None:
            # 对每个小区取所有波束的平均信道增益
            avg_channel_gains = np.mean(self.channel_gains, axis=0)
            return avg_channel_gains
        else:
            # 如果还未计算信道增益，返回默认值
            return np.ones(self.num_cells)


class DynamicQueueModel(QueueModel):
    """
    动态队列模型类 - 继承自基础队列模型，增加动态流量特性
    """
    
    def __init__(self, *args, **kwargs):
        # 添加动态流量参数
        self.traffic_pattern = kwargs.pop('traffic_pattern', 'mixed')  # 'uniform', 'hotspot', 'bursty', 'mixed'
        self.burst_probability = kwargs.pop('burst_probability', 0.1)
        self.burst_intensity = kwargs.pop('burst_intensity', 5.0)
        self.hotspot_cells = kwargs.pop('hotspot_cells', None)
        self.user_mobility_enabled = kwargs.pop('user_mobility_enabled', True)
        
        super().__init__(*args, **kwargs)
        
        # 初始化动态状态
        self.current_pattern = self.traffic_pattern
        self.burst_state = np.zeros(self.num_cells)
        self.hotspot_intensity = np.ones(self.num_cells)
        self.user_density = np.ones(self.num_cells)  # 相对用户密度
        
        # 流量模式切换参数
        self.pattern_switch_prob = 0.01  # 流量模式切换概率
        self.time_since_pattern_change = 0
        
        if self.hotspot_cells is None:
            # 随机选择一些小区作为热点
            self.hotspot_cells = np.random.choice(
                self.num_cells, 
                size=max(1, self.num_cells // 4), 
                replace=False
            )
    
    def generate_traffic(self):
        """生成动态业务请求"""
        # 更新流量模式
        self._update_traffic_pattern()
        
        # 更新用户分布
        if self.user_mobility_enabled:
            self._update_user_distribution()
        
        # 更新突发状态
        self._update_burst_state()
        
        # 生成基础流量
        arrivals = super().generate_traffic()
        
        # 应用动态效应
        arrivals = self._apply_dynamic_effects(arrivals)
        
        return arrivals
    
    def _update_traffic_pattern(self):
        """更新流量模式"""
        self.time_since_pattern_change += 1
        
        # 随机切换流量模式
        if np.random.random() < self.pattern_switch_prob:
            patterns = ['uniform', 'hotspot', 'bursty', 'mixed']
            self.current_pattern = np.random.choice(patterns)
            self.time_since_pattern_change = 0
    
    def _update_user_distribution(self):
        """更新用户分布（模拟用户移动）"""
        # 用户密度的慢变化（模拟用户移动）
        mobility_factor = 0.02  # 移动性系数
        
        for cell_idx in range(self.num_cells):
            # 添加随机游走
            change = np.random.normal(0, mobility_factor)
            self.user_density[cell_idx] += change
            
            # 确保密度在合理范围内
            self.user_density[cell_idx] = np.clip(self.user_density[cell_idx], 0.1, 3.0)
        
        # 归一化用户密度（保持总用户数相对稳定）
        total_density = np.sum(self.user_density)
        self.user_density = self.user_density / total_density * self.num_cells
    
    def _update_burst_state(self):
        """更新突发状态"""
        for cell_idx in range(self.num_cells):
            # 突发开始
            if self.burst_state[cell_idx] == 0 and np.random.random() < self.burst_probability:
                self.burst_state[cell_idx] = np.random.randint(5, 20)  # 突发持续时间
            
            # 突发结束
            if self.burst_state[cell_idx] > 0:
                self.burst_state[cell_idx] -= 1
    
    def _apply_dynamic_effects(self, base_arrivals):
        """应用动态效应到基础流量"""
        arrivals = base_arrivals.copy()
        
        for cell_idx in range(self.num_cells):
            # 应用用户密度效应
            arrivals[cell_idx] *= self.user_density[cell_idx]
            
            # 应用流量模式效应
            if self.current_pattern == 'hotspot' and cell_idx in self.hotspot_cells:
                arrivals[cell_idx] *= 3.0  # 热点小区流量增加
            elif self.current_pattern == 'uniform':
                # 均匀分布，不做额外处理
                pass
            elif self.current_pattern == 'mixed':
                # 混合模式：部分小区高流量，部分小区低流量
                if cell_idx % 3 == 0:
                    arrivals[cell_idx] *= 2.0
                elif cell_idx % 3 == 1:
                    arrivals[cell_idx] *= 0.5
            
            # 应用突发效应
            if self.burst_state[cell_idx] > 0:
                arrivals[cell_idx] *= self.burst_intensity
        return arrivals
    
    def get_traffic_state_info(self):
        """获取流量状态信息"""
        return {
            'current_pattern': self.current_pattern,
            'burst_state': self.burst_state.copy(),
            'user_density': self.user_density.copy(),
            'hotspot_cells': self.hotspot_cells.copy()
        }
    
    def get_current_traffic(self):
        """获取当前小区的业务请求量"""
        # 生成当前时刻的流量
        current_arrivals = self.generate_traffic()
        return current_arrivals


class DynamicSatelliteEnv(gym.Env):
    """
    动态卫星跳波束环境 - 增强版环境，包含多种动态特性
    
    新增特性：
    1. 动态信道：衰落、阴影、天气影响、卫星移动
    2. 动态流量：突发流量、热点小区、用户移动
    3. 多目标优化：吞吐量、延迟、公平性、能效
    4. 非平稳环境：流量模式切换、信道条件变化
    """
    
    def __init__(self, 
                 num_beams: int = 4, 
                 num_cells: int = 19, 
                 total_power_dbw: float = 39.0,
                 total_bandwidth_mhz: float = 200.0,
                 satellite_height_km: float = 550.0,
                 max_queue_size: int = 100,
                 packet_ttl: int = 12,
                 traffic_intensity: float = 0.7,
                 # 新增动态参数
                 enable_channel_dynamics: bool = True,
                 enable_traffic_dynamics: bool = True,
                 enable_weather: bool = True,
                 enable_mobility: bool = True,
                 reward_weights: Dict[str, float] = None):
        """
        初始化动态卫星环境
        """
        super(DynamicSatelliteEnv, self).__init__()
        
        # 基础参数
        self.num_beams = num_beams
        self.num_cells = num_cells
        self.total_power_dbw = total_power_dbw
        self.total_power_linear = 10 ** (total_power_dbw / 10)
        self.total_bandwidth_mhz = total_bandwidth_mhz
        self.satellite_height_km = satellite_height_km
        self.max_queue_size = max_queue_size
        self.packet_ttl = packet_ttl
        self.traffic_intensity = traffic_intensity
        
        # 动态特性开关
        self.enable_channel_dynamics = enable_channel_dynamics
        self.enable_traffic_dynamics = enable_traffic_dynamics
        self.enable_weather = enable_weather
        self.enable_mobility = enable_mobility
        
        # 奖励权重
        if reward_weights is None:
            self.reward_weights = {
                'throughput': 0.4,
                'delay': 0.25,
                'fairness': 0.2,
                'energy_efficiency': 0.15
            }
        else:
            self.reward_weights = reward_weights
        
        # 创建动态信道模型
        self.channel_model = DynamicChannelModel(
            num_beams=num_beams,
            num_cells=num_cells,
            satellite_height_km=satellite_height_km,
            fading_enabled=enable_channel_dynamics,
            shadowing_enabled=enable_channel_dynamics,
            weather_enabled=enable_weather,
            satellite_movement_enabled=enable_mobility
        )
        
        # 创建动态队列模型
        self.queue_model = DynamicQueueModel(
            num_cells=num_cells,
            max_queue_size=max_queue_size,
            packet_ttl=packet_ttl,
            traffic_intensity=traffic_intensity,
            user_mobility_enabled=enable_mobility,
            traffic_pattern='mixed' if enable_traffic_dynamics else 'uniform'
        )
          # 扩展观测空间（包含更多状态信息）
        # 观测包括：信道增益(4*19=76)、队列状态(19)、队列长度(19)、历史性能(2)、环境状态(3)
        obs_dim = num_beams * num_cells + num_cells + num_cells + 2 + 3  # 76+19+19+2+3=119
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 动作空间保持不变
        self.beam_allocation_space = spaces.MultiDiscrete([num_cells] * num_beams)
        self.power_allocation_space = spaces.Box(
            low=np.zeros(num_beams),
            high=np.ones(num_beams),
            dtype=np.float32
        )
        self.action_space = spaces.Dict({
            'beam_allocation': self.beam_allocation_space,
            'power_allocation': self.power_allocation_space
        })
        
        # 初始化状态
        self.channel_gains = None
        self.traffic_requests = None
        self.current_step = 0
        self.max_steps = 2000  # 增加episode长度以观察动态性
        
        # 性能历史（用于计算奖励）
        self.performance_history = {
            'throughput': [],
            'delay': [],
            'fairness': [],
            'energy_efficiency': [],
            'served_per_cell': []
        }
        
        # 环境状态跟踪
        self.env_state = {
            'weather_severity': 0.0,
            'traffic_burstiness': 0.0,
            'channel_quality': 1.0        }
    
    def reset(self):
        """重置环境"""
        self.current_step = 0
        
        # 重置模型
        self.channel_model.reset()
        self.queue_model.reset()
        
        # 预先生成一些初始流量，避免第一步队列为空
        initial_traffic = self.queue_model.generate_traffic()
        
        # 获取初始状态
        self.channel_gains = self.channel_model.get_channel_gains()
        self.traffic_requests = self.queue_model.get_queue_state()
        
        # 重置性能历史
        self.performance_history = {
            'throughput': [],
            'delay': [],
            'fairness': [],
            'energy_efficiency': [],
            'served_per_cell': []
        }
        
        # 重置环境状态
        self.env_state = {
            'weather_severity': 0.0,
            'traffic_burstiness': 0.0,
            'channel_quality': 1.0
        }
        
        return self._get_observation()
    
    def step(self, action: Dict):
        """执行一步环境交互"""
        # 解析动作
        beam_allocation = np.array(action['beam_allocation']).flatten()
        power_allocation = np.array(action['power_allocation']).flatten()
        
        # 确保beam_allocation长度正确
        if len(beam_allocation) != self.num_beams:
            # 如果长度不正确，使用随机分配或重复/截断
            if len(beam_allocation) == 0:
                beam_allocation = np.random.randint(0, self.num_cells, size=self.num_beams)
            elif len(beam_allocation) < self.num_beams:
                # 重复到正确长度
                beam_allocation = np.tile(beam_allocation, (self.num_beams // len(beam_allocation) + 1))[:self.num_beams]
            else:
                # 截断到正确长度
                beam_allocation = beam_allocation[:self.num_beams]
        
        # 确保beam_allocation值在有效范围内
        beam_allocation = np.clip(beam_allocation, 0, self.num_cells - 1).astype(int)
        
        # 确保动作有效性
        if len(power_allocation) != self.num_beams:
            power_allocation = np.ones(self.num_beams) / self.num_beams
        
        power_allocation = np.maximum(power_allocation, 0.0)
        power_sum = np.sum(power_allocation)
        power_allocation = power_allocation / power_sum if power_sum > 0 else np.ones(self.num_beams) / self.num_beams
        
        # 计算实际功率分配
        actual_power = power_allocation * self.total_power_linear
        
        # 计算信噪比和数据率
        sinr, data_rates = self._calculate_sinr_and_data_rates(beam_allocation, actual_power)
          # 更新队列状态
        served_packets, dropped_packets, avg_delay = self.queue_model.update(data_rates, sinr)
        
        # 生成新的业务请求（动态流量）
        request_data = self.queue_model.generate_traffic()
        
        # 更新信道状态（动态信道）
        self.channel_model.update()
        self.channel_gains = self.channel_model.get_channel_gains()
        
        # 获取新的队列状态
        self.traffic_requests = self.queue_model.get_queue_state()
        # print("queue: ", self.traffic_requests)
        
        # 更新环境状态
        self._update_environment_state()
        
        # 计算多目标奖励
        reward = self._calculate_multi_objective_reward(
            served_packets, avg_delay, dropped_packets, 
            data_rates, power_allocation
        )
        
        # 更新性能历史
        self._update_performance_history(served_packets, avg_delay, data_rates, power_allocation)
        
        # 更新步数
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # 构建信息字典
        info = self._get_info_dict(served_packets, avg_delay, dropped_packets, sinr, data_rates)
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """获取增强的观测信息"""
        # 基础观测：信道增益和队列状态
        channel_flat = self.channel_gains.flatten()
        queue_state = self.traffic_requests.flatten()
        
        # 队列长度信息
        queue_lengths = self.queue_model.get_queue_lengths()
        
        # 历史性能信息（最近几步的平均）
        window_size = min(10, len(self.performance_history['throughput']))
        if window_size > 0:
            avg_throughput = np.mean(self.performance_history['throughput'][-window_size:])
            avg_delay = np.mean(self.performance_history['delay'][-window_size:])
        else:
            avg_throughput = 0.0
            avg_delay = 0.0
        
        # 环境状态信息
        env_state_vector = np.array([
            self.env_state['weather_severity'],
            self.env_state['traffic_burstiness'], 
            self.env_state['channel_quality']
        ])
        
        # 组合所有观测信息
        observation = np.concatenate([
            channel_flat,
            queue_state,
            queue_lengths,
            np.array([avg_throughput, avg_delay]),
            env_state_vector
        ])
        
        return observation.astype(np.float32)
    
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
    
    def _calculate_multi_objective_reward(self, served_packets, avg_delay, dropped_packets, 
                                        data_rates, power_allocation):
        """计算多目标奖励函数"""
        # 吞吐量奖励（归一化）
        total_throughput = np.sum(served_packets)
        max_possible_throughput = self.num_cells * 10  # 假设最大每个小区10个包
        throughput_reward = total_throughput / max_possible_throughput
        
        # 延迟奖励（越小越好）
        max_delay = self.packet_ttl
        delay_reward = 1.0 - min(avg_delay / max_delay, 1.0)
        
        # 公平性奖励（Jain's fairness index）
        if np.sum(served_packets) > 0:
            fairness_reward = np.sum(served_packets)**2 / (self.num_cells * np.sum(served_packets**2))
        else:
            fairness_reward = 1.0
        
        # 能效奖励（吞吐量/功率）
        total_power = np.sum(power_allocation)
        if total_power > 0 and total_throughput > 0:
            energy_efficiency = total_throughput / total_power
            # 归一化能效
            max_efficiency = max_possible_throughput / (self.num_beams * 0.1)  # 假设最小功率0.1
            energy_efficiency_reward = min(energy_efficiency / max_efficiency, 1.0)
        else:
            energy_efficiency_reward = 0.0
        
        # 组合奖励
        reward = (
            self.reward_weights['throughput'] * throughput_reward +
            self.reward_weights['delay'] * delay_reward +
            self.reward_weights['fairness'] * fairness_reward +
            self.reward_weights['energy_efficiency'] * energy_efficiency_reward
        )
        
        # 添加惩罚项
        if dropped_packets > 0:
            reward -= 0.1 * dropped_packets / max_possible_throughput
        
        return reward
    
    def _update_performance_history(self, served_packets, avg_delay, data_rates, power_allocation):
        """更新性能历史"""
        total_throughput = np.sum(served_packets)
        
        # 计算公平性
        if np.sum(served_packets) > 0:
            fairness = np.sum(served_packets)**2 / (self.num_cells * np.sum(served_packets**2))
        else:
            fairness = 1.0
        
        # 计算能效
        total_power = np.sum(power_allocation)
        if total_power > 0 and total_throughput > 0:
            energy_efficiency = total_throughput / total_power
        else:
            energy_efficiency = 0.0
        
        # 添加到历史
        self.performance_history['throughput'].append(total_throughput)
        self.performance_history['delay'].append(avg_delay)
        self.performance_history['fairness'].append(fairness)
        self.performance_history['energy_efficiency'].append(energy_efficiency)
        self.performance_history['served_per_cell'].append(served_packets.copy())
        
        # 保持历史长度
        max_history = 100
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def _update_environment_state(self):
        """更新环境状态"""
        # 获取信道状态信息
        channel_info = self.channel_model.get_channel_state_info()
        
        # 计算天气严重程度
        if self.enable_weather:
            avg_weather_attenuation = np.mean(channel_info['weather_attenuation'])
            self.env_state['weather_severity'] = min(avg_weather_attenuation / 10.0, 1.0)
        
        # 获取流量状态信息
        traffic_info = self.queue_model.get_traffic_state_info()
        
        # 计算流量突发性
        burst_ratio = np.sum(traffic_info['burst_state'] > 0) / self.num_cells
        self.env_state['traffic_burstiness'] = burst_ratio
        
        # 计算信道质量
        avg_fading = np.mean(channel_info['fading_state'])
        self.env_state['channel_quality'] = min(avg_fading, 2.0) / 2.0
    
    def _get_info_dict(self, served_packets, avg_delay, dropped_packets, sinr, data_rates):
        """构建详细的信息字典"""
        # 基础性能指标
        info = {
            'throughput': np.sum(served_packets),
            'delay': avg_delay,
            'packet_loss': dropped_packets,
            'sinr': sinr.copy(),
            'data_rates': data_rates.copy(),
            'served_per_cell': served_packets.copy()
        }
        
        # 动态环境状态
        info.update(self.env_state)
        
        # 性能分析
        if len(self.performance_history['throughput']) > 0:
            info['throughput_trend'] = (
                self.performance_history['throughput'][-1] - 
                np.mean(self.performance_history['throughput'][-10:])
            ) if len(self.performance_history['throughput']) >= 10 else 0
        
        # 公平性指标
        if np.sum(served_packets) > 0:
            info['fairness_index'] = np.sum(served_packets)**2 / (self.num_cells * np.sum(served_packets**2))
        else:
            info['fairness_index'] = 1.0
        
        return info
    
    def get_environment_complexity_score(self):
        """获取当前环境复杂度评分"""
        complexity_score = 0.0
        
        # 信道复杂度
        if self.enable_channel_dynamics:
            complexity_score += 0.3
        
        # 流量复杂度
        if self.enable_traffic_dynamics:
            complexity_score += 0.3
        
        # 天气复杂度
        if self.enable_weather:
            complexity_score += 0.2 * self.env_state['weather_severity']
        
        # 移动性复杂度
        if self.enable_mobility:
            complexity_score += 0.2
        
        return complexity_score
    
    def set_complexity_level(self, level: str):
        """设置环境复杂度等级"""
        if level == 'simple':
            self.enable_channel_dynamics = False
            self.enable_traffic_dynamics = False
            self.enable_weather = False
            self.enable_mobility = False
        elif level == 'medium':
            self.enable_channel_dynamics = True
            self.enable_traffic_dynamics = True
            self.enable_weather = False
            self.enable_mobility = False
        elif level == 'complex':
            self.enable_channel_dynamics = True
            self.enable_traffic_dynamics = True
            self.enable_weather = True
            self.enable_mobility = True
        
        # 重新初始化模型
        self.__init__(
            num_beams=self.num_beams,
            num_cells=self.num_cells,
            total_power_dbw=self.total_power_dbw,
            total_bandwidth_mhz=self.total_bandwidth_mhz,
            satellite_height_km=self.satellite_height_km,
            max_queue_size=self.max_queue_size,
            packet_ttl=self.packet_ttl,
            traffic_intensity=self.traffic_intensity,
            enable_channel_dynamics=self.enable_channel_dynamics,
            enable_traffic_dynamics=self.enable_traffic_dynamics,
            enable_weather=self.enable_weather,
            enable_mobility=self.enable_mobility,
            reward_weights=self.reward_weights
        )
