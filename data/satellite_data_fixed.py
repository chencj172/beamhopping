import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class SatelliteDataFixed:
    """
    固定卫星位置数据类
    
    用于生成固定位置的卫星数据
    """
    
    def __init__(self, 
                 satellite_height_km: float = 1000.0):
        """
        初始化固定卫星位置数据类
        
        Args:
            satellite_height_km: 卫星高度(km)
        """
        self.satellite_height_km = satellite_height_km
        
        # 地球半径 (km)
        self.earth_radius_km = 6371.0
        
        # 生成固定的卫星位置（正对赤道上空）
        self.satellite_position = self._generate_satellite_position()
    
    def _generate_satellite_position(self):
        """
        生成固定的卫星位置
        
        Returns:
            satellite_position: 卫星位置，shape=(3,)，单位：km
        """
        # 卫星位于北美西部上空
        # 北美西部大致位置：纬度约40°N，经度约120°W
        lat_rad = np.radians(40.0)  # 北纬40度
        lon_rad = np.radians(-120.0)  # 西经120度
        
        # 计算卫星在地心坐标系中的位置
        r = (self.earth_radius_km + self.satellite_height_km) * 1000
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        return np.array([x, y, z])
    
    def get_position(self, time_step=None):
        """
        获取卫星位置（由于位置固定，time_step参数不起作用，但保留以兼容原接口）
        
        Args:
            time_step: 时间步索引（不使用）
        
        Returns:
            position: 卫星位置，shape=(3,)，单位：km
        """
        return self.satellite_position
    
    def plot_orbit(self):
        """
        绘制卫星位置（由于位置固定，只显示一个点）
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
        
        # 绘制卫星位置
        ax.scatter(self.satellite_position[0], 
                  self.satellite_position[1], 
                  self.satellite_position[2], 
                  color='red', s=100, label='Satellite')
        
        # 设置坐标轴标签
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        
        # 设置坐标轴范围
        limit = self.earth_radius_km * 1.2
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        
        # 设置标题
        ax.set_title(f'Fixed Satellite Position (Height: {self.satellite_height_km} km)')
        
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
        np.save(file_path, self.satellite_position)
        
        print(f"固定卫星位置数据已保存到：{file_path}")
    
    def save_to_csv(self, file_path: str):
        """
        保存卫星位置数据到CSV文件
        
        Args:
            file_path: 文件路径
        """
        # 创建DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'X': [self.satellite_position[0]],
            'Y': [self.satellite_position[1]],
            'Z': [self.satellite_position[2]]
        })
        
        # 保存到CSV
        df.to_csv(file_path, index=False)
        
        print(f"固定卫星位置数据已保存到CSV文件：{file_path}")
    
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
        satellite_position = np.load(file_path)
        
        # 创建对象
        satellite_data = SatelliteDataFixed()
        satellite_data.satellite_position = satellite_position
        
        return satellite_data


# 示例用法
if __name__ == "__main__":
    # 创建固定卫星位置数据
    satellite_data = SatelliteDataFixed(
        satellite_height_km=1000.0
    )
    
    # 绘制卫星位置
    satellite_data.plot_orbit()
    
    # 保存数据
    satellite_data.save_to_file("satellite_position_fixed.npy")