import numpy as np
from typing import List, Tuple, Dict


class ChannelModel:
    """
    信道模型类
    
    用于模拟卫星与地面小区之间的信道增益
    信道增益计算为：发射天线增益 * 路径损耗 * 接收天线增益
    """
    
    def __init__(self, 
                 num_beams: int = 4, 
                 num_cells: int = 19, 
                 satellite_height_km: float = 1000.0,
                 frequency_ghz: float = 20.0,  # Ka频段典型频率
                 tx_antenna_gain_db: float = 30.0,  # 发射天线增益
                 rx_antenna_gain_db: float = 20.0,  # 接收天线增益
                 channel_variation_std: float = 0.1):  # 信道变化标准差
        """
        初始化信道模型
        
        Args:
            num_beams: 波束数量
            num_cells: 小区数量
            satellite_height_km: 卫星高度(km)
            frequency_ghz: 工作频率(GHz)
            tx_antenna_gain_db: 发射天线增益(dB)
            rx_antenna_gain_db: 接收天线增益(dB)
            channel_variation_std: 信道变化标准差
        """
        self.num_beams = num_beams
        self.num_cells = num_cells
        self.satellite_height_km = satellite_height_km
        self.frequency_ghz = frequency_ghz
        self.tx_antenna_gain_db = tx_antenna_gain_db
        self.rx_antenna_gain_db = rx_antenna_gain_db
        self.channel_variation_std = channel_variation_std
        
        # 光速 (m/s)
        self.c = 3e8
        
        # 波长 (m)
        self.wavelength = self.c / (self.frequency_ghz * 1e9)
        
        # 初始化卫星位置
        self.satellite_position = np.array([0, 0, satellite_height_km * 1000])  # 单位：米
        
        # 初始化小区位置
        self.cell_positions = self._initialize_cell_positions()
        
        # 初始化信道增益矩阵
        self.channel_gains = np.zeros((num_beams, num_cells))
        
        # 初始化波束方向
        self.beam_directions = self._initialize_beam_directions()
        
        # 计算初始信道增益
        self._update_channel_gains()
    
    def reset(self):
        """
        重置信道模型
        """
        # 重新初始化波束方向
        self.beam_directions = self._initialize_beam_directions()
        
        # 重新计算信道增益
        self._update_channel_gains()
    
    def update(self):
        """
        更新信道状态
        """
        # 添加随机变化
        self._add_channel_variation()
    
    def get_channel_gains(self):
        """
        获取当前信道增益矩阵
        
        Returns:
            channel_gains: 信道增益矩阵 (num_beams x num_cells)
        """
        return self.channel_gains
    
    def _initialize_cell_positions(self):
        """
        初始化小区位置
        
        Returns:
            cell_positions: 小区位置数组，shape=(num_cells, 3)，单位：米
        """
        # 创建六边形网格布局
        # 小区半径 (km)
        cell_radius_km = 50
        
        # 转换为米
        cell_radius = cell_radius_km * 1000
        
        # 小区中心位置
        positions = []
        
        # 中心小区
        positions.append([0, 0, 0])
        
        # 第一环（6个小区）
        for i in range(6):
            angle = i * 60  # 角度，单位：度
            angle_rad = np.radians(angle)  # 转换为弧度
            x = cell_radius * np.cos(angle_rad)
            y = cell_radius * np.sin(angle_rad)
            positions.append([x, y, 0])
        
        # 第二环（12个小区）
        for i in range(12):
            angle = i * 30  # 角度，单位：度
            angle_rad = np.radians(angle)  # 转换为弧度
            x = 2 * cell_radius * np.cos(angle_rad)
            y = 2 * cell_radius * np.sin(angle_rad)
            positions.append([x, y, 0])
        
        # 确保小区数量正确
        positions = positions[:self.num_cells]
        
        return np.array(positions)
    
    def _initialize_beam_directions(self):
        """
        初始化波束方向
        
        Returns:
            beam_directions: 波束方向数组，shape=(num_beams, 3)
        """
        # 随机选择小区作为波束指向的目标
        target_cells = np.random.choice(self.num_cells, self.num_beams, replace=False)
        
        # 计算从卫星到目标小区的方向向量
        beam_directions = []
        for cell_idx in target_cells:
            cell_position = self.cell_positions[cell_idx]
            direction = cell_position - self.satellite_position
            # 归一化方向向量
            direction = direction / np.linalg.norm(direction)
            beam_directions.append(direction)
        
        return np.array(beam_directions)
    
    def _update_channel_gains(self):
        """
        更新信道增益矩阵
        """
        # 计算每个波束到每个小区的信道增益
        for beam_idx in range(self.num_beams):
            for cell_idx in range(self.num_cells):
                # 计算卫星到小区的距离
                cell_position = self.cell_positions[cell_idx]
                distance = np.linalg.norm(self.satellite_position - cell_position)
                
                # 计算路径损耗 (dB)
                path_loss_db = self._calculate_path_loss(distance)
                
                # 计算天线增益 (dB)
                antenna_gain_db = self._calculate_antenna_gain(beam_idx, cell_idx)
                
                # 计算总信道增益 (dB)
                channel_gain_db = self.tx_antenna_gain_db + antenna_gain_db + self.rx_antenna_gain_db - path_loss_db
                
                # 转换为线性值
                self.channel_gains[beam_idx, cell_idx] = 10 ** (channel_gain_db / 10)
    
    def _calculate_path_loss(self, distance):
        """
        计算路径损耗
        
        Args:
            distance: 距离 (m)
        
        Returns:
            path_loss_db: 路径损耗 (dB)
        """
        # 自由空间路径损耗模型
        # PL(dB) = 20*log10(4*pi*d/lambda)
        path_loss_db = 20 * np.log10(4 * np.pi * distance / self.wavelength)
        
        return path_loss_db
    
    def _calculate_antenna_gain(self, beam_idx, cell_idx):
        """
        计算天线增益
        
        Args:
            beam_idx: 波束索引
            cell_idx: 小区索引
        
        Returns:
            antenna_gain_db: 天线增益 (dB)
        """
        # 计算波束方向与小区方向的夹角
        beam_direction = self.beam_directions[beam_idx]
        cell_position = self.cell_positions[cell_idx]

        # 计算从卫星到小区的方向向量
        cell_direction = cell_position - self.satellite_position
        cell_direction = cell_direction / np.linalg.norm(cell_direction)
        
        # 计算夹角 (弧度)
        cos_angle = np.dot(beam_direction, cell_direction)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在[-1, 1]范围内
        angle_rad = np.arccos(cos_angle)
        
        # 转换为角度
        angle_deg = np.degrees(angle_rad)
        
        # 使用简化的天线增益模型
        # 波束宽度 (度)
        beamwidth = 8.0
        
        # 计算天线增益
        if angle_deg <= beamwidth / 2:
            # 主瓣
            antenna_gain_db = 0  # 相对增益，主瓣为0dB
        else:
            # 旁瓣，使用简化模型
            antenna_gain_db = -12 * (angle_deg / beamwidth) ** 2
            # 限制最小增益
            antenna_gain_db = max(antenna_gain_db, -30)
        
        return antenna_gain_db
    
    def _add_channel_variation(self):
        """
        添加信道随机变化
        """
        # 添加对数正态分布的随机变化
        variation_db = np.random.normal(0, self.channel_variation_std, self.channel_gains.shape)
        
        # 转换为线性值并应用
        variation_linear = 10 ** (variation_db / 10)
        self.channel_gains = self.channel_gains * variation_linear