import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional


class Visualization:
    """
    可视化工具类
    
    用于可视化系统性能和结果
    """
    
    @staticmethod
    def plot_learning_curve(rewards: List[float], avg_window: int = 100, title: str = 'Learning Curve'):
        """
        绘制学习曲线
        
        Args:
            rewards: 奖励列表
            avg_window: 平均窗口大小
            title: 图表标题
        """
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制原始奖励
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
        
        # 计算移动平均
        if len(rewards) >= avg_window:
            avg_rewards = []
            for i in range(len(rewards) - avg_window + 1):
                avg_rewards.append(np.mean(rewards[i:i+avg_window]))
            plt.plot(range(avg_window-1, len(rewards)), avg_rewards, color='red', label=f'Moving Average ({avg_window})')
        
        # 设置坐标轴标签
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # 设置标题
        plt.title(title)
        
        # 添加图例
        plt.legend()
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_performance_metrics(metrics: Dict[str, List[float]], title: str = 'Performance Metrics'):
        """
        绘制性能指标
        
        Args:
            metrics: 性能指标字典，键为指标名称，值为指标值列表
            title: 图表标题
        """
        # 创建图形
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)), sharex=True)
        
        # 如果只有一个指标，将axes转换为列表
        if len(metrics) == 1:
            axes = [axes]
        
        # 绘制每个指标
        for i, (metric_name, metric_values) in enumerate(metrics.items()):
            axes[i].plot(metric_values)
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴标签
        axes[-1].set_xlabel('Episode')
        
        # 设置标题
        fig.suptitle(title)
        
        # 显示图形
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()
    
    @staticmethod
    def plot_beam_allocation(beam_allocation: np.ndarray, cell_positions: np.ndarray, 
                            power_allocation: Optional[np.ndarray] = None, 
                            traffic_demand: Optional[np.ndarray] = None,
                            title: str = 'Beam Allocation'):
        """
        绘制波束分配情况
        
        Args:
            beam_allocation: 波束分配数组，shape=(num_beams,)
            cell_positions: 小区位置数组，shape=(num_cells, 2)
            power_allocation: 功率分配数组，shape=(num_beams,)，可选
            traffic_demand: 业务请求数组，shape=(num_cells,)，可选
            title: 图表标题
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 获取小区数量
        num_cells = len(cell_positions)
        
        # 获取波束数量
        num_beams = len(beam_allocation)
        
        # 创建颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, num_beams))
        
        # 绘制小区
        for i, pos in enumerate(cell_positions):
            # 创建六边形
            hex_radius = 0.8
            hexagon = RegularPolygon((pos[0], pos[1]), numVertices=6, radius=hex_radius, 
                                     orientation=0, edgecolor='black', alpha=0.5)
            
            # 确定小区颜色
            cell_color = 'lightgray'  # 默认颜色（未分配波束）
            for beam_idx, cell_idx in enumerate(beam_allocation):
                if cell_idx == i:
                    cell_color = colors[beam_idx]
                    break
            
            # 设置填充颜色
            hexagon.set_facecolor(cell_color)
            
            # 添加到图形
            plt.gca().add_patch(hexagon)
            
            # 添加小区索引标签
            label_text = f'{i+1}'  # 小区索引
            
            # 如果提供了业务请求，添加到标签
            if traffic_demand is not None:
                label_text += f'\n{traffic_demand[i]:.1f}'  # 业务请求
            
            plt.text(pos[0], pos[1], label_text, 
                     ha='center', va='center', color='black', fontsize=10, fontweight='bold')
        
        # 添加波束标签
        for beam_idx, cell_idx in enumerate(beam_allocation):
            if cell_idx < num_cells:  # 确保小区索引有效
                pos = cell_positions[cell_idx]
                
                # 波束标签
                beam_label = f'B{beam_idx+1}'  # 波束索引
                
                # 如果提供了功率分配，添加到标签
                if power_allocation is not None:
                    beam_label += f'\n{power_allocation[beam_idx]:.2f}'  # 功率分配
                
                plt.text(pos[0], pos[1] + 0.5, beam_label, 
                         ha='center', va='center', color=colors[beam_idx], 
                         fontsize=12, fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[beam_idx]))
        
        # 创建图例
        legend_elements = []
        for beam_idx in range(num_beams):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=colors[beam_idx], markersize=10, 
                                            label=f'Beam {beam_idx+1}'))
        plt.legend(handles=legend_elements, loc='upper right')
        
        # 设置坐标轴范围
        plt.xlim([np.min(cell_positions[:, 0]) - 2, np.max(cell_positions[:, 0]) + 2])
        plt.ylim([np.min(cell_positions[:, 1]) - 2, np.max(cell_positions[:, 1]) + 2])
        
        # 设置坐标轴相等比例
        plt.axis('equal')
        
        # 设置标题
        plt.title(title)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_sinr_heatmap(sinr: np.ndarray, cell_positions: np.ndarray, 
                         beam_allocation: Optional[np.ndarray] = None,
                         title: str = 'SINR Heatmap'):
        """
        绘制信噪比热力图
        
        Args:
            sinr: 信噪比数组，shape=(num_cells,)
            cell_positions: 小区位置数组，shape=(num_cells, 2)
            beam_allocation: 波束分配数组，shape=(num_beams,)，可选
            title: 图表标题
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 获取小区数量
        num_cells = len(cell_positions)
        
        # 获取信噪比范围
        sinr_min = np.min(sinr)
        sinr_max = np.max(sinr)
        
        # 创建颜色映射
        norm = mcolors.Normalize(vmin=sinr_min, vmax=sinr_max)
        cmap = plt.cm.jet
        
        # 绘制小区
        for i, pos in enumerate(cell_positions):
            # 创建六边形
            hex_radius = 0.8
            hexagon = RegularPolygon((pos[0], pos[1]), numVertices=6, radius=hex_radius, 
                                     orientation=0, edgecolor='black')
            
            # 设置填充颜色（根据信噪比）
            hexagon.set_facecolor(cmap(norm(sinr[i])))
            
            # 添加到图形
            plt.gca().add_patch(hexagon)
            
            # 添加小区索引和信噪比标签
            plt.text(pos[0], pos[1], f'{i+1}\n{sinr[i]:.1f} dB', 
                     ha='center', va='center', color='black', fontsize=10, fontweight='bold')
        
        # 如果提供了波束分配，添加波束标签
        if beam_allocation is not None:
            # 创建颜色映射
            beam_colors = plt.cm.tab10(np.linspace(0, 1, len(beam_allocation)))
            
            for beam_idx, cell_idx in enumerate(beam_allocation):
                if cell_idx < num_cells:  # 确保小区索引有效
                    pos = cell_positions[cell_idx]
                    plt.text(pos[0], pos[1] + 0.5, f'B{beam_idx+1}', 
                             ha='center', va='center', color=beam_colors[beam_idx], 
                             fontsize=12, fontweight='bold', 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor=beam_colors[beam_idx]))
        
        # 添加颜色条
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                           orientation='vertical', shrink=0.8)
        cbar.set_label('SINR (dB)')
        
        # 设置坐标轴范围
        plt.xlim([np.min(cell_positions[:, 0]) - 2, np.max(cell_positions[:, 0]) + 2])
        plt.ylim([np.min(cell_positions[:, 1]) - 2, np.max(cell_positions[:, 1]) + 2])
        
        # 设置坐标轴相等比例
        plt.axis('equal')
        
        # 设置标题
        plt.title(title)
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_queue_status(queue_lengths: np.ndarray, cell_positions: np.ndarray, 
                         max_queue_size: int, title: str = 'Queue Status'):
        """
        绘制队列状态
        
        Args:
            queue_lengths: 队列长度数组，shape=(num_cells,)
            cell_positions: 小区位置数组，shape=(num_cells, 2)
            max_queue_size: 队列最大容量
            title: 图表标题
        """
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 获取小区数量
        num_cells = len(cell_positions)
        
        # 创建颜色映射
        norm = mcolors.Normalize(vmin=0, vmax=max_queue_size)
        cmap = plt.cm.YlOrRd  # 黄-橙-红色映射
        
        # 绘制小区
        for i, pos in enumerate(cell_positions):
            # 创建六边形
            hex_radius = 0.8
            hexagon = RegularPolygon((pos[0], pos[1]), numVertices=6, radius=hex_radius, 
                                     orientation=0, edgecolor='black')
            
            # 设置填充颜色（根据队列长度）
            hexagon.set_facecolor(cmap(norm(queue_lengths[i])))
            
            # 添加到图形
            plt.gca().add_patch(hexagon)
            
            # 添加小区索引和队列长度标签
            plt.text(pos[0], pos[1], f'{i+1}\n{queue_lengths[i]}/{max_queue_size}', 
                     ha='center', va='center', color='black', fontsize=10, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                           orientation='vertical', shrink=0.8)
        cbar.set_label('Queue Length')
        
        # 设置坐标轴范围
        plt.xlim([np.min(cell_positions[:, 0]) - 2, np.max(cell_positions[:, 0]) + 2])
        plt.ylim([np.min(cell_positions[:, 1]) - 2, np.max(cell_positions[:, 1]) + 2])
        
        # 设置坐标轴相等比例
        plt.axis('equal')
        
        # 设置标题
        plt.title(title)
        
        # 显示图形
        plt.tight_layout()
        plt.show()