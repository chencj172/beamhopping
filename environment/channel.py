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
        self.satellite_position = np.array([-2650896.795413222,-4591487.935277223,4448733.046640539])
        
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
        # 首先尝试从CSV文件加载小区位置
        try:
            cell_positions = self._load_cell_positions_from_csv()
            print("成功从CSV文件加载小区位置")
            return cell_positions
        except Exception as e:
            print(f"从CSV文件加载小区位置失败: {e}")
            print("使用默认的六边形网格布局")
            return self._generate_default_cell_positions()
    
    def _load_cell_positions_from_csv(self, csv_path="data_dir/cell_positions_fixed.csv"):
        """
        从CSV文件加载小区位置
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            cell_positions: 小区位置数组，shape=(num_cells, 3)，单位：米
        """
        import os
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        try:
            # 尝试使用pandas读取（更方便）
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # 检查列名
            required_columns = ['X', 'Y', 'Z']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV文件缺少必需的列: {col}")
            
            # 提取位置数据
            positions = df[['X', 'Y', 'Z']].values
            
        except ImportError:
            # 如果pandas不可用，使用numpy读取
            print("pandas不可用，使用numpy读取CSV文件")
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)  # 跳过标题行
            
            # 假设CSV格式为: Cell_Index,X,Y,Z
            if data.shape[1] < 4:
                raise ValueError("CSV文件格式不正确，应包含至少4列")
            
            # 提取X,Y,Z列（索引1,2,3）
            positions = data[:, 1:4]
        
        # 确保小区数量匹配
        if len(positions) < self.num_cells:
            raise ValueError(f"CSV文件中的小区数量({len(positions)})少于所需数量({self.num_cells})")
        
        # 取前num_cells个小区
        positions = positions[:self.num_cells]
        
        print(f"从CSV文件加载了{len(positions)}个小区位置")
        return np.array(positions)
    
    def _generate_default_cell_positions(self):
        """
        生成默认的六边形网格小区位置
        
        Returns:
            cell_positions: 小区位置数组，shape=(num_cells, 3)，单位：米
        """
        # 创建六边形网格布局
        # 小区半径 (km)
        cell_radius_km = 60
        
        # 转换为米
        cell_radius = cell_radius_km * 1000
        
        # 小区中心位置
        positions = []
        
        # 中心小区
        positions.append([40, -120, 0])
        
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
        # 自由空间路径损耗模型 (Free Space Path Loss, FSPL)
        # PL(dB) = 20*log10(4*pi*d*f/c)
        # 其中：d是距离(m)，f是频率(Hz)，c是光速(m/s)
        
        frequency_hz = self.frequency_ghz * 1e9  # 转换为Hz
        
        # 计算路径损耗
        path_loss_db = 20 * np.log10(4 * np.pi * distance * frequency_hz / self.c)
        
        # 确保路径损耗为正值且在合理范围内
        path_loss_db = max(path_loss_db, 32.45)  # 最小自由空间损耗(1km@1GHz)
        
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
        direction_norm = np.linalg.norm(cell_direction)
        
        # 避免除零错误
        if direction_norm < 1e-10:
            return -30  # 返回最小增益
            
        cell_direction = cell_direction / direction_norm
        
        # 计算夹角 (弧度)
        cos_angle = np.dot(beam_direction, cell_direction)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在[-1, 1]范围内
        angle_rad = np.arccos(cos_angle)
        
        # 转换为角度
        angle_deg = np.degrees(angle_rad)
        
        # 使用改进的天线增益模型
        # 3dB波束宽度 (度)
        beamwidth_3db = 8.0
        
        # 计算天线增益
        if angle_deg <= beamwidth_3db / 2:
            # 主瓣内，使用抛物线模型
            antenna_gain_db = -3 * (angle_deg / (beamwidth_3db / 2)) ** 2
        else:
            # 旁瓣，使用衰减模型
            # 按照ITU-R建议，旁瓣衰减通常为 -12 * (θ/θ_3dB)^2
            antenna_gain_db = -12 * (angle_deg / beamwidth_3db) ** 2
            # 限制最小增益，避免过度衰减
            antenna_gain_db = max(antenna_gain_db, -40)
        
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
    
    def reload_cell_positions(self, csv_path=None):
        """
        重新加载小区位置
        
        Args:
            csv_path: CSV文件路径，如果为None则使用默认路径
        """
        if csv_path is None:
            csv_path = "data_dir/cell_positions_fixed.csv"
        
        try:
            self.cell_positions = self._load_cell_positions_from_csv(csv_path)
            # 重新计算信道增益
            self._update_channel_gains()
            print("小区位置重新加载成功")
        except Exception as e:
            print(f"重新加载小区位置失败: {e}")
            print("保持当前小区位置不变")
    
    def get_cell_positions(self):
        """
        获取当前小区位置
        
        Returns:
            cell_positions: 小区位置数组，shape=(num_cells, 3)，单位：米
        """
        return self.cell_positions.copy()