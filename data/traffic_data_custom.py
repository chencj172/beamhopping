import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple


class TrafficDataCustom:
    """
    自定义业务请求数据生成类
    
    根据指定的小区业务请求量生成数据
    """
    
    def __init__(self, 
                 num_cells: int = 19,
                 num_time_steps: int = 86400,  # 24小时，每秒一个时间步
                 daily_pattern: bool = True,
                 random_seed: int = None):
        """
        初始化业务请求数据生成类
        
        Args:
            num_cells: 小区数量
            num_time_steps: 时间步数量（默认为86400，表示24小时，每秒一个时间步）
            daily_pattern: 是否使用日变化模式
            random_seed: 随机种子
        """
        self.num_cells = num_cells
        self.num_time_steps = num_time_steps
        self.daily_pattern = daily_pattern
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 根据图片中的数据设置每个小区的基础泊松参数
        # 小区ID从1开始，数组索引从0开始
        self.cell_lambda = {
            1: 40,   # 小区1的业务请求量
            2: 50,   # 小区2的业务请求量
            3: 50,   # 小区3的业务请求量
            4: 30,   # 小区4的业务请求量
            5: 90,   # 小区5的业务请求量
            6: 50,   # 小区6的业务请求量
            7: 60,   # 小区7的业务请求量
            8: 10,   # 小区8的业务请求量
            9: 60,   # 小区9的业务请求量
            10: 100,  # 小区10的业务请求量
            11: 100,  # 小区11的业务请求量
            12: 20,   # 小区12的业务请求量
            13: 70,   # 小区13的业务请求量
            14: 60,   # 小区14的业务请求量
            15: 10,   # 小区15的业务请求量
            16: 20,   # 小区16的业务请求量
            17: 70,   # 小区17的业务请求量
            18: 50,   # 小区18的业务请求量
            19: 90    # 小区19的业务请求量
        }
        
        # 生成业务请求数据
        self.traffic_data = self._generate_traffic_data()
    
    def _generate_traffic_data(self):
        """
        生成业务请求数据
        
        Returns:
            traffic_data: 业务请求数据，shape=(num_time_steps, num_cells)，单位：Mbps
        """
        # 初始化业务请求数据数组
        traffic_data = np.zeros((self.num_time_steps, self.num_cells))
        
        # 获取每个小区的基础泊松参数
        base_lambda = np.array([self.cell_lambda[i+1] for i in range(self.num_cells)])
        
        # 生成每个时间步的业务请求数据
        for t in range(self.num_time_steps):
            # 基础泊松参数
            lambda_values = base_lambda.copy()
            
            # 添加日变化模式
            if self.daily_pattern:
                # 计算日变化系数
                daily_factor = self._calculate_daily_factor(t)
                
                # 应用日变化系数
                lambda_values = lambda_values * daily_factor
            
            # 使用泊松分布生成请求量
            traffic = np.random.poisson(lambda_values)
            
            # 存储业务请求数据
            traffic_data[t] = traffic
        
        return traffic_data
    
    def _calculate_daily_factor(self, time_step: int):
        """
        计算日变化系数
        
        Args:
            time_step: 时间步索引
        
        Returns:
            daily_factor: 日变化系数，shape=(num_cells,)
        """
        # 初始化日变化系数
        daily_factor = np.ones(self.num_cells)
        
        # 如果不使用日变化模式，则返回全1系数
        if not self.daily_pattern:
            return daily_factor
        
        # 计算一天中的小时数（每秒一个时间步）
        hour_of_day = (time_step % 86400) / 3600.0
        
        # 根据一天中的不同时段设置不同的系数
        # 早高峰：7-10点
        # 工作时间：10-17点
        # 晚高峰：17-22点
        # 夜间：22-7点
        
        # 为每个小区计算日变化系数
        for i in range(self.num_cells):
            # 基础系数
            if 0 <= hour_of_day < 7:  # 夜间
                factor = 0.3 + 0.2 * np.random.random()
            elif 7 <= hour_of_day < 10:  # 早高峰
                factor = 0.7 + 0.6 * np.random.random()
            elif 10 <= hour_of_day < 17:  # 工作时间
                factor = 0.6 + 0.4 * np.random.random()
            elif 17 <= hour_of_day < 22:  # 晚高峰
                factor = 0.8 + 0.4 * np.random.random()
            else:  # 22-24点
                factor = 0.4 + 0.3 * np.random.random()
            
            # 存储日变化系数
            daily_factor[i] = factor
        
        return daily_factor
    
    def get_traffic_data(self, time_step: int):
        """
        获取指定时间步的业务请求数据
        
        Args:
            time_step: 时间步索引
        
        Returns:
            traffic: 业务请求数据，shape=(num_cells,)，单位：Mbps
        """
        if time_step < 0 or time_step >= self.num_time_steps:
            raise ValueError(f"时间步索引超出范围：{time_step}，应在[0, {self.num_time_steps-1}]范围内")
        
        return self.traffic_data[time_step]
    
    def get_all_traffic_data(self):
        """
        获取所有时间步的业务请求数据
        
        Returns:
            traffic_data: 业务请求数据，shape=(num_time_steps, num_cells)，单位：Mbps
        """
        return self.traffic_data
    
    def plot_traffic_patterns(self, cell_indices=None):
        """
        绘制业务请求模式
        
        Args:
            cell_indices: 要绘制的小区索引列表，如果为None则绘制所有小区
        """
        # 如果未指定小区索引，则绘制所有小区
        if cell_indices is None:
            cell_indices = list(range(self.num_cells))
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 时间轴（小时）
        time_hours = np.arange(self.num_time_steps) / 3600.0  # 每秒一个时间步
        
        # 绘制每个小区的业务请求模式
        for cell_idx in cell_indices:
            # 为了避免图表过于密集，可以采样数据点（每10分钟一个点）
            sample_interval = 600  # 10分钟 = 600秒
            sampled_time = time_hours[::sample_interval]
            sampled_traffic = self.traffic_data[::sample_interval, cell_idx]
            plt.plot(sampled_time, sampled_traffic, label=f'Cell {cell_idx+1}')
        
        # 设置坐标轴标签
        plt.xlabel('Time (hours)')
        plt.ylabel('Traffic Demand (Mbps)')
        
        # 设置标题
        plt.title('Traffic Demand Pattern (Custom Distribution)')
        
        # 添加图例
        plt.legend()
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    def plot_traffic_heatmap(self, time_step: int):
        """
        绘制指定时间步的业务请求热力图
        
        Args:
            time_step: 时间步索引
        """
        # 获取指定时间步的业务请求数据
        traffic = self.get_traffic_data(time_step)
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        plt.bar(range(1, self.num_cells + 1), traffic)
        
        # 设置坐标轴标签
        plt.xlabel('Cell ID')
        plt.ylabel('Traffic Demand (Mbps)')
        
        # 设置标题
        hour_of_day = (time_step % 86400) / 3600.0
        plt.title(f'Traffic Demand Heatmap at Hour {hour_of_day:.2f}')
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图形
        plt.tight_layout()
        plt.show()