import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class CellData:
    """
    小区位置数据生成类
    
    用于生成卫星覆盖的小区位置数据
    """
    
    def __init__(self, 
                 num_cells: int = 19,
                 cell_radius_km: float = 50.0,
                 satellite_height_km: float = 1000.0):
        """
        初始化小区位置数据生成类
        
        Args:
            num_cells: 小区数量
            cell_radius_km: 小区半径(km)
            satellite_height_km: 卫星高度(km)
        """
        self.num_cells = num_cells
        self.cell_radius_km = cell_radius_km
        self.satellite_height_km = satellite_height_km
        
        # 地球半径 (km)
        self.earth_radius_km = 6371.0
        
        # 生成小区位置数据
        self.cell_positions, self.cell_indices = self._generate_cell_positions()
    
    def _generate_cell_positions(self):
        """
        生成小区位置数据
        
        Returns:
            cell_positions: 小区位置数据，shape=(num_cells, 3)，单位：km
            cell_indices: 小区索引，用于标识小区
        """
        # 创建六边形网格布局
        positions = []
        indices = []
        
        # 中心小区
        positions.append([0, 0, 0])
        indices.append(1)  # 中心小区索引为1
        
        # 第一环（6个小区）
        for i in range(6):
            angle = i * 60  # 角度，单位：度
            angle_rad = np.radians(angle)  # 转换为弧度
            x = self.cell_radius_km * np.cos(angle_rad)
            y = self.cell_radius_km * np.sin(angle_rad)
            positions.append([x, y, 0])
            indices.append(i + 2)  # 第一环小区索引从2开始
        
        # 第二环（12个小区）
        for i in range(12):
            angle = i * 30  # 角度，单位：度
            angle_rad = np.radians(angle)  # 转换为弧度
            x = 2 * self.cell_radius_km * np.cos(angle_rad)
            y = 2 * self.cell_radius_km * np.sin(angle_rad)
            positions.append([x, y, 0])
            indices.append(i + 8)  # 第二环小区索引从8开始
        
        # 确保小区数量正确
        positions = positions[:self.num_cells]
        indices = indices[:self.num_cells]
        
        # 将小区位置投影到地球表面
        earth_positions = self._project_to_earth_surface(positions)
        
        return np.array(earth_positions), np.array(indices)
    
    def _project_to_earth_surface(self, positions):
        """
        将平面小区位置投影到地球表面
        
        Args:
            positions: 平面小区位置列表
        
        Returns:
            earth_positions: 地球表面小区位置列表
        """
        earth_positions = []
        
        for pos in positions:
            # 计算从原点到小区中心的距离
            distance = np.sqrt(pos[0]**2 + pos[1]**2)
            
            # 计算地球表面上的角度（弧度）
            angle = distance / self.earth_radius_km
            
            # 计算地球表面上的位置
            x = self.earth_radius_km * np.sin(angle) * pos[0] / distance if distance > 0 else 0
            y = self.earth_radius_km * np.sin(angle) * pos[1] / distance if distance > 0 else 0
            z = self.earth_radius_km * np.cos(angle)
            
            earth_positions.append([x, y, z])
        
        return earth_positions
    
    def get_cell_position(self, cell_idx: int):
        """
        获取指定小区的位置
        
        Args:
            cell_idx: 小区索引
        
        Returns:
            position: 小区位置，shape=(3,)，单位：km
        """
        if cell_idx < 0 or cell_idx >= self.num_cells:
            raise ValueError(f"小区索引超出范围：{cell_idx}，应在[0, {self.num_cells-1}]范围内")
        
        return self.cell_positions[cell_idx]
    
    def get_all_cell_positions(self):
        """
        获取所有小区的位置
        
        Returns:
            positions: 小区位置数组，shape=(num_cells, 3)，单位：km
        """
        return self.cell_positions
    
    def get_cell_indices(self):
        """
        获取小区索引
        
        Returns:
            indices: 小区索引数组
        """
        return self.cell_indices
    
    def plot_cells(self, satellite_position=None):
        """
        绘制小区分布
        
        Args:
            satellite_position: 卫星位置，shape=(3,)，单位：km
        """
        # 创建图形
        fig = plt.figure(figsize=(12, 10))
        
        # 3D视图
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 绘制地球
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.earth_radius_km * np.cos(u) * np.sin(v)
        y = self.earth_radius_km * np.sin(u) * np.sin(v)
        z = self.earth_radius_km * np.cos(v)
        ax1.plot_surface(x, y, z, color='blue', alpha=0.3)
        
        # 绘制小区
        for i, pos in enumerate(self.cell_positions):
            ax1.scatter(pos[0], pos[1], pos[2], color='green', s=100)
            ax1.text(pos[0], pos[1], pos[2], str(self.cell_indices[i]), color='black')
        
        # 绘制卫星（如果提供了位置）
        if satellite_position is not None:
            ax1.scatter(satellite_position[0], satellite_position[1], satellite_position[2], 
                       color='red', s=200, label='Satellite')
            
            # 绘制从卫星到每个小区的连线
            for pos in self.cell_positions:
                ax1.plot([satellite_position[0], pos[0]], 
                         [satellite_position[1], pos[1]], 
                         [satellite_position[2], pos[2]], 'k--', alpha=0.3)
        
        # 设置坐标轴标签
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_zlabel('Z (km)')
        
        # 设置坐标轴范围
        limit = self.earth_radius_km * 1.2
        ax1.set_xlim([-limit, limit])
        ax1.set_ylim([-limit, limit])
        ax1.set_zlim([-limit, limit])
        
        # 设置标题
        ax1.set_title('3D View of Cells on Earth Surface')
        
        # 添加图例
        if satellite_position is not None:
            ax1.legend()
        
        # 2D视图（六边形网格）
        ax2 = fig.add_subplot(122)
        
        # 绘制六边形网格
        for i in range(self.num_cells):
            # 创建六边形
            angles = np.linspace(0, 2*np.pi, 7)[:-1]
            hex_x = self.cell_radius_km * np.cos(angles)
            hex_y = self.cell_radius_km * np.sin(angles)
            
            # 计算六边形中心
            if i == 0:  # 中心小区
                center_x, center_y = 0, 0
            elif i < 7:  # 第一环
                angle = (i - 1) * 60  # 角度，单位：度
                angle_rad = np.radians(angle)  # 转换为弧度
                center_x = self.cell_radius_km * np.cos(angle_rad)
                center_y = self.cell_radius_km * np.sin(angle_rad)
            else:  # 第二环
                angle = (i - 7) * 30  # 角度，单位：度
                angle_rad = np.radians(angle)  # 转换为弧度
                center_x = 2 * self.cell_radius_km * np.cos(angle_rad)
                center_y = 2 * self.cell_radius_km * np.sin(angle_rad)
            
            # 平移六边形到中心位置
            hex_x = hex_x + center_x
            hex_y = hex_y + center_y
            
            # 绘制六边形
            ax2.fill(hex_x, hex_y, 'green', alpha=0.3, edgecolor='black')
            
            # 添加小区索引标签
            ax2.text(center_x, center_y, str(self.cell_indices[i]), 
                     ha='center', va='center', color='red', fontsize=12, fontweight='bold')
        
        # 设置坐标轴标签
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        
        # 设置坐标轴范围
        limit = 2.5 * self.cell_radius_km
        ax2.set_xlim([-limit, limit])
        ax2.set_ylim([-limit, limit])
        
        # 设置标题
        ax2.set_title('2D View of Hexagonal Cell Grid')
        
        # 设置坐标轴相等比例
        ax2.set_aspect('equal')
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    def save_to_file(self, file_path: str):
        """
        保存小区位置数据到文件
        
        Args:
            file_path: 文件路径
        """
        # 保存数据
        data = {
            'positions': self.cell_positions,
            'indices': self.cell_indices
        }
        np.save(file_path, data)
        
        print(f"小区位置数据已保存到：{file_path}")
    
    @staticmethod
    def load_from_file(file_path: str):
        """
        从文件加载小区位置数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            cell_data: 小区位置数据对象
        """
        # 加载数据
        data = np.load(file_path, allow_pickle=True).item()
        
        # 创建对象
        cell_data = CellData()
        cell_data.cell_positions = data['positions']
        cell_data.cell_indices = data['indices']
        cell_data.num_cells = len(data['positions'])
        
        return cell_data


# 示例用法
if __name__ == "__main__":
    # 创建小区位置数据生成器
    cell_data = CellData(
        num_cells=19,
        cell_radius_km=50.0,
        satellite_height_km=1000.0
    )
    
    # 绘制小区分布
    satellite_position = np.array([0, 0, 1000 + 6371.0])  # 卫星位置
    cell_data.plot_cells(satellite_position)
    
    # 保存数据
    cell_data.save_to_file("cell_positions.npy")