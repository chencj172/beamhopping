import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple


class TrafficDataPoisson:
    """
    业务请求数据生成类（泊松分布）
    
    用于生成服从泊松分布的业务请求数据
    """
    
    def __init__(self, 
                 num_cells: int = 19,
                 num_time_steps: int = 86400,  # 24小时，每秒一个时间步
                 min_lambda: float = 10.0,
                 max_lambda: float = 100.0,
                 daily_pattern: bool = True,
                 random_seed: int = None):
        """
        初始化业务请求数据生成类
        
        Args:
            num_cells: 小区数量
            num_time_steps: 时间步数量（默认为86400，表示24小时，每秒一个时间步）
            min_lambda: 最小泊松参数λ
            max_lambda: 最大泊松参数λ
            daily_pattern: 是否使用日变化模式
            random_seed: 随机种子
        """
        self.num_cells = num_cells
        self.num_time_steps = num_time_steps
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.daily_pattern = daily_pattern
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
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
        
        # 为每个小区生成基础泊松参数
        base_lambda = np.random.uniform(
            self.min_lambda,
            self.max_lambda,
            size=self.num_cells
        )
        
        # 生成每个时间步的业务请求数据
        for t in range(self.num_time_steps):
            # 基础泊松参数
            lambda_values = base_lambda.copy()
            
            # 添加日变化模式
            if self.daily_pattern:
                # 计算一天中的时间（小时）
                hour_of_day = (t % 1440) / 60.0  # 假设每分钟一个时间步
                
                # 日变化系数（模拟工作时间高峰和夜间低谷）
                daily_factor = self._calculate_daily_factor(hour_of_day)
                
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
        plt.title('Traffic Demand Pattern (Poisson Distribution)')
        
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
        
        # 创建六边形网格布局
        positions = []
        
        # 中心小区
        positions.append([0, 0])
        
        # 第一环（6个小区）
        for i in range(6):
            angle = i * 60  # 角度，单位：度
            angle_rad = np.radians(angle)  # 转换为弧度
            x = np.cos(angle_rad)
            y = np.sin(angle_rad)
            positions.append([x, y])
        
        # 第二环（12个小区）
        for i in range(12):
            angle = i * 30  # 角度，单位：度
            angle_rad = np.radians(angle)  # 转换为弧度
            x = 2 * np.cos(angle_rad)
            y = 2 * np.sin(angle_rad)
            positions.append([x, y])
        
        # 确保小区数量正确
        positions = positions[:self.num_cells]
        
        # 绘制六边形网格
        for i, pos in enumerate(positions):
            # 创建六边形
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = 0.8 * np.cos(angles) + pos[0]
            hex_y = 0.8 * np.sin(angles) + pos[1]
            
            # 根据业务量确定颜色
            color_intensity = traffic[i] / (self.max_lambda * 2)  # 归一化到[0, 1]范围
            color_intensity = np.clip(color_intensity, 0, 1)  # 限制在[0, 1]范围内
            
            # 绘制六边形
            plt.fill(hex_x, hex_y, color=plt.cm.jet(color_intensity), edgecolor='black')
            
            # 添加小区索引和业务量标签
            plt.text(pos[0], pos[1], f'{i+1}\n{traffic[i]:.1f}', 
                     ha='center', va='center', color='black', fontsize=10, fontweight='bold')
        
        # 设置坐标轴范围
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        
        # 设置坐标轴相等比例
        plt.axis('equal')
        
        # 关闭坐标轴
        plt.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.jet), 
                           orientation='vertical', shrink=0.8)
        cbar.set_label('Traffic Demand (Mbps)')
        
        # 设置标题
        hour = (time_step % 86400) / 3600.0
        plt.title(f'Traffic Demand Heatmap at Hour {hour:.1f} (Poisson Distribution)')
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    def save_to_file(self, file_path: str):
        """
        保存业务请求数据到文件
        
        Args:
            file_path: 文件路径
        """
        # 保存数据
        np.save(file_path, self.traffic_data)
        
        print(f"业务请求数据已保存到：{file_path}")
    
    def save_to_csv(self, file_path: str):
        """
        保存业务请求数据到CSV文件
        
        Args:
            file_path: 文件路径
        """
        # 创建DataFrame
        columns = [f'Cell_{i+1}' for i in range(self.num_cells)]
        df = pd.DataFrame(self.traffic_data, columns=columns)
        
        # 添加时间列（秒级时间步）
        df['Time'] = [f'{(t // 3600):02d}:{((t % 3600) // 60):02d}:{(t % 60):02d}' for t in range(self.num_time_steps)]
        
        # 添加时间戳列（秒）
        df['Timestamp'] = list(range(self.num_time_steps))
        
        # 重新排列列，使时间列在最前面
        cols = ['Time', 'Timestamp'] + columns
        df = df[cols]
        
        # 保存到CSV
        df.to_csv(file_path, index=False)
        
        print(f"业务请求数据已保存到CSV文件：{file_path}")
    
    @staticmethod
    def load_from_file(file_path: str):
        """
        从文件加载业务请求数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            traffic_data: 业务请求数据对象
        """
        # 加载数据
        traffic_data_array = np.load(file_path)
        
        # 创建对象
        traffic_data = TrafficDataPoisson()
        traffic_data.traffic_data = traffic_data_array
        traffic_data.num_time_steps, traffic_data.num_cells = traffic_data_array.shape
        
        return traffic_data
    
    @staticmethod
    def load_from_csv(file_path: str):
        """
        从CSV文件加载业务请求数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            traffic_data: 业务请求数据对象
        """
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 提取小区数据（排除时间列）
        cell_columns = [col for col in df.columns if col != 'Time']
        traffic_data_array = df[cell_columns].values
        
        # 创建对象
        traffic_data = TrafficDataPoisson()
        traffic_data.traffic_data = traffic_data_array
        traffic_data.num_time_steps, traffic_data.num_cells = traffic_data_array.shape
        
        return traffic_data


# 示例用法
if __name__ == "__main__":
    # 创建业务请求数据生成器
    traffic_data = TrafficDataPoisson(
        num_cells=19,
        num_time_steps=1440,  # 24小时，每分钟一个时间步
        min_lambda=10.0,
        max_lambda=100.0,
        daily_pattern=True,
        random_seed=42
    )
    
    # 绘制业务请求模式
    traffic_data.plot_traffic_patterns(cell_indices=[0, 1, 2, 3, 4])
    
    # 绘制业务请求热力图
    traffic_data.plot_traffic_heatmap(time_step=540)  # 9:00
    
    # 保存数据
    traffic_data.save_to_file("traffic_data_poisson.npy")
    
    # 保存为CSV格式
    traffic_data.save_to_csv("traffic_data_poisson.csv")