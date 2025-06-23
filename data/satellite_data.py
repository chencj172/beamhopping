import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class SatelliteData:
    """
    卫星位置数据生成类
    
    用于生成卫星位置数据
    """
    
    def __init__(self, 
                 satellite_height_km: float = 1000.0,
                 inclination_deg: float = 98.0,  # 近极轨道倾角
                 period_minutes: float = 105.0,  # 轨道周期
                 simulation_time_hours: float = 24.0):  # 模拟时间
        """
        初始化卫星位置数据生成类
        
        Args:
            satellite_height_km: 卫星高度(km)
            inclination_deg: 轨道倾角(度)
            period_minutes: 轨道周期(分钟)
            simulation_time_hours: 模拟时间(小时)
        """
        self.satellite_height_km = satellite_height_km
        self.inclination_deg = inclination_deg
        self.period_minutes = period_minutes
        self.simulation_time_hours = simulation_time_hours
        
        # 地球半径 (km)
        self.earth_radius_km = 6371.0
        
        # 轨道半径 (km)
        self.orbit_radius_km = self.earth_radius_km + satellite_height_km
        
        # 轨道周期 (秒)
        self.period_seconds = period_minutes * 60.0
        
        # 模拟时间 (秒)
        self.simulation_time_seconds = simulation_time_hours * 3600.0
        
        # 时间步长 (秒)
        self.time_step_seconds = 60.0  # 1分钟
        
        # 时间步数
        self.num_time_steps = int(self.simulation_time_seconds / self.time_step_seconds)
        
        # 生成卫星位置数据
        self.satellite_positions = self._generate_satellite_positions()
    
    def _generate_satellite_positions(self):
        """
        生成卫星位置数据
        
        Returns:
            satellite_positions: 卫星位置数据，shape=(num_time_steps, 3)，单位：km
        """
        # 初始化卫星位置数组
        satellite_positions = np.zeros((self.num_time_steps, 3))
        
        # 轨道倾角 (弧度)
        inclination_rad = np.radians(self.inclination_deg)
        
        # 生成每个时间步的卫星位置
        for t in range(self.num_time_steps):
            # 当前时间 (秒)
            current_time = t * self.time_step_seconds
            
            # 计算轨道角度 (弧度)
            orbit_angle = 2 * np.pi * (current_time % self.period_seconds) / self.period_seconds
            
            # 计算卫星在轨道平面内的位置
            x_orbit = self.orbit_radius_km * np.cos(orbit_angle)
            y_orbit = self.orbit_radius_km * np.sin(orbit_angle)
            
            # 考虑轨道倾角，计算卫星在地心坐标系中的位置
            x = x_orbit
            y = y_orbit * np.cos(inclination_rad)
            z = y_orbit * np.sin(inclination_rad)
            
            # 存储卫星位置
            satellite_positions[t] = [x, y, z]
        
        return satellite_positions
    
    def get_satellite_position(self, time_step: int):
        """
        获取指定时间步的卫星位置
        
        Args:
            time_step: 时间步索引
        
        Returns:
            position: 卫星位置，shape=(3,)，单位：km
        """
        if time_step < 0 or time_step >= self.num_time_steps:
            raise ValueError(f"时间步索引超出范围：{time_step}，应在[0, {self.num_time_steps-1}]范围内")
        
        return self.satellite_positions[time_step]
    
    def get_all_satellite_positions(self):
        """
        获取所有时间步的卫星位置
        
        Returns:
            positions: 卫星位置数组，shape=(num_time_steps, 3)，单位：km
        """
        return self.satellite_positions
    
    def plot_orbit(self, num_points: int = 100):
        """
        绘制卫星轨道
        
        Args:
            num_points: 轨道上的点数
        """
        # 创建图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地球
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.earth_radius_km * np.cos(u) * np.sin(v)
        y = self.earth_radius_km * np.sin(u) * np.sin(v)
        z = self.earth_radius_km * np.cos(v)
        ax.plot_surface(x, y, z, color='blue', alpha=0.3)
        
        # 绘制卫星轨道
        orbit_points = self.satellite_positions[::self.num_time_steps//num_points]
        ax.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], 'r-', linewidth=2)
        
        # 绘制卫星当前位置
        current_pos = self.satellite_positions[0]
        ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='red', s=100, label='Satellite')
        
        # 设置坐标轴标签
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        
        # 设置坐标轴范围
        limit = self.orbit_radius_km * 1.2
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        
        # 设置标题
        ax.set_title(f'Satellite Orbit (Height: {self.satellite_height_km} km, Inclination: {self.inclination_deg}°)')
        
        # 添加图例
        ax.legend()
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    def save_to_file(self, file_path: str):
        """
        保存卫星位置数据到文件
        
        Args:
            file_path: 文件路径
        """
        # 保存数据
        np.save(file_path, self.satellite_positions)
        
        print(f"卫星位置数据已保存到：{file_path}")
    
    @staticmethod
    def load_from_file(file_path: str):
        """
        从文件加载卫星位置数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            satellite_data: 卫星位置数据对象
        """
        # 加载数据
        satellite_positions = np.load(file_path)
        
        # 创建对象
        satellite_data = SatelliteData()
        satellite_data.satellite_positions = satellite_positions
        satellite_data.num_time_steps = len(satellite_positions)
        
        return satellite_data


# 示例用法
if __name__ == "__main__":
    # 创建卫星位置数据生成器
    satellite_data = SatelliteData(
        satellite_height_km=1000.0,
        inclination_deg=98.0,
        period_minutes=105.0,
        simulation_time_hours=24.0
    )
    
    # 绘制卫星轨道
    satellite_data.plot_orbit()
    
    # 保存数据
    satellite_data.save_to_file("satellite_positions.npy")