import numpy as np
from typing import List, Tuple, Dict


class QueueModel:
    """
    队列模型类
    
    用于模拟各小区的业务请求队列
    每个小区都有一个队列，业务请求包会先插入到该队列中
    服务的业务包会从队列中清除，业务包有TTL限制
    """
    
    def __init__(self, 
                 num_cells: int = 19, 
                 max_queue_size: int = 500,
                 packet_ttl: int = 15,
                 min_arrival_rate: float = 0.1,  # 最小到达率（数据包/时隙）
                 max_arrival_rate: float = 1.0,  # 最大到达率（数据包/时隙）
                 packet_size_kb: float = 100.0,  # 数据包大小 (KB) - 调整为更合理的大小
                 traffic_intensity: float = 0.8):  # 流量强度因子
        """
        初始化队列模型
        
        Args:
            num_cells: 小区数量
            max_queue_size: 队列最大容量
            packet_ttl: 数据包生存时间(时隙)
            min_arrival_rate: 最小到达率（数据包/时隙）
            max_arrival_rate: 最大到达率（数据包/时隙）
            packet_size_kb: 数据包大小 (KB)
            traffic_intensity: 流量强度因子 (0.0-1.0)，影响总体到达率
        """
        self.num_cells = num_cells
        self.max_queue_size = max_queue_size
        self.packet_ttl = packet_ttl
        self.min_arrival_rate = min_arrival_rate
        self.max_arrival_rate = max_arrival_rate
        self.packet_size_kb = packet_size_kb
        self.traffic_intensity = traffic_intensity

        # 初始化队列
        self.queues = [[] for _ in range(num_cells)]
        
        # 初始化到达率
        self.arrival_rates = self._initialize_arrival_rates()
        
        # 初始化性能指标
        self.total_arrivals = 0
        self.total_served = 0
        self.total_dropped = 0
        self.total_delay = 0
    
    def reset(self):
        """
        重置队列模型
        """
        # 清空队列
        self.queues = [[] for _ in range(self.num_cells)]
        
        # 重新初始化到达率
        self.arrival_rates = self._initialize_arrival_rates()
        
        # 重置性能指标
        self.total_arrivals = 0
        self.total_served = 0
        self.total_dropped = 0
        self.total_delay = 0
    
    def update(self, data_rates, sinr_values=None):
        """
        更新队列状态
        
        Args:
            data_rates: 数据率数组，单位：bits/s
            sinr_values: 信噪比数组（可选，用于调试）
        
        Returns:
            served_packets: 服务的数据包数量
            dropped_packets: 丢弃的数据包数量
            avg_delay: 平均延迟
        """
        # 初始化统计变量
        served_packets = np.zeros(self.num_cells)
        dropped_packets = 0
        total_delay = 0
        total_packets = 0

        # 更新每个小区的队列
        for cell_idx in range(self.num_cells):
            # 计算可以服务的数据包数量
            # 数据率单位：bits/s，数据包大小单位：KB
            # 假设时隙长度为1秒
            packet_size_bits = self.packet_size_kb * 1000 * 8  # KB转换为bits
            max_serve = int(data_rates[cell_idx] / packet_size_bits) if packet_size_bits > 0 else 0
            # if max_serve > 0:
                # print(max_serve)
            
            # 服务队列中的数据包
            served = 0
            while served < max_serve and self.queues[cell_idx]:
                # 获取队首数据包
                packet = self.queues[cell_idx][0]
                
                # 从队列中移除数据包
                self.queues[cell_idx].pop(0)
                
                # 更新统计信息
                served += 1
                total_delay += packet['age']
                total_packets += 1
            
            # 记录服务的数据包数量
            served_packets[cell_idx] = served
            self.total_served += served

            # 更新队列中剩余数据包的年龄
            dropped_in_cell = 0
            i = 0
            while i < len(self.queues[cell_idx]):
                packet = self.queues[cell_idx][i]
                packet['age'] += 1
                
                # 检查是否超过TTL
                if packet['age'] > self.packet_ttl:
                    # 丢弃数据包
                    self.queues[cell_idx].pop(i)
                    dropped_in_cell += 1
                else:
                    i += 1
            
            # 更新丢弃的数据包数量
            dropped_packets += dropped_in_cell
            self.total_dropped += dropped_in_cell
        
        # 计算平均延迟
        avg_delay = total_delay / total_packets if total_packets > 0 else 0
        
        return served_packets, dropped_packets, avg_delay
    
    def generate_traffic(self):
        """
        生成新的业务请求
        
        Returns:
            arrivals: 新生成的数据包数量
        """
        # 初始化到达数据包数量
        arrivals = np.zeros(self.num_cells)
        
        # 更新到达率（模拟业务变化）
        self._update_arrival_rates()
        
        # 为每个小区生成新的数据包
        for cell_idx in range(self.num_cells):
            # 使用泊松分布生成到达数据包数量
            arrival_count = np.random.poisson(self.arrival_rates[cell_idx]) * 200
            # 限制队列容量
            available_space = self.max_queue_size - len(self.queues[cell_idx])
            arrival_count = min(arrival_count, available_space)
            
            # 生成新的数据包
            for _ in range(arrival_count):
                # 创建数据包
                packet = {
                    'size': self.packet_size_kb,  # 数据包大小 (KB)
                    'age': 0  # 初始年龄
                }
                
                # 添加到队列
                self.queues[cell_idx].append(packet)
            
            # 记录到达数据包数量
            arrivals[cell_idx] = arrival_count
            self.total_arrivals += arrival_count
        
        return arrivals
    
    def get_queue_state(self):
        """
        获取队列状态
        
        Returns:
            queue_state: 队列状态数组，表示每个小区的队列长度
        """
        # 计算每个小区的队列长度
        queue_state = np.array([len(queue) for queue in self.queues])
        
        return queue_state
    
    def get_queue_lengths(self):
        """
        获取队列长度数组
        
        Returns:
            queue_lengths: 队列长度数组        """
        return np.array([len(queue) for queue in self.queues])
    
    def _initialize_arrival_rates(self):
        """
        初始化到达率 - 基于实际地理位置的热点分布
        
        Returns:
            arrival_rates: 到达率数组
        """
        # 基于实际地理位置坐标分析的中心辐射式热点分布
        arrival_rates = np.zeros(self.num_cells)
        
        # 基于地理位置分析的分层结构（小区索引转换为数组索引需要减1）
        # 核心区域：距离中心最近的小区
        core_cells = [0]  # 小区1 -> 索引0，真正的地理中心
        
        # 内环区域：紧邻中心的小区
        inner_ring = [3, 1, 2, 6, 4, 11]  # 小区4,2,3,7,5,12 -> 索引3,1,2,6,4,11
        
        # 中环区域：中等距离的小区
        middle_ring = [15, 7, 8, 9, 12, 10]  # 小区16,8,9,10,13,11 -> 索引15,7,8,9,12,10
        
        # 外环区域：边缘小区
        outer_ring = [13, 14, 16, 18, 17]  # 小区14,15,17,19,18 -> 索引13,14,16,18,17
        
        # 其余小区设为外围区域
        remaining_cells = [5]  # 小区6 -> 索引5
        
        # 初始化基础到达率（最低流量）
        base_rate = self.min_arrival_rate + 0.1 * (self.max_arrival_rate - self.min_arrival_rate)
        arrival_rates.fill(base_rate)
        
        # 设置核心区域的最高到达率
        for cell_idx in core_cells:
            if cell_idx < self.num_cells:
                arrival_rates[cell_idx] = self.max_arrival_rate  # 100% 最大流量
        
        # 设置内环区域的高到达率
        for cell_idx in inner_ring:
            if cell_idx < self.num_cells:
                arrival_rates[cell_idx] = self.max_arrival_rate * 0.8  # 80% 最大流量
        
        # 设置中环区域的中等到达率
        for cell_idx in middle_ring:
            if cell_idx < self.num_cells:
                arrival_rates[cell_idx] = self.max_arrival_rate * 0.5  # 50% 最大流量
        
        # 设置外环区域的较低到达率
        for cell_idx in outer_ring:
            if cell_idx < self.num_cells:
                arrival_rates[cell_idx] = self.max_arrival_rate * 0.3  # 30% 最大流量
        
        # 设置其余区域的低到达率
        for cell_idx in remaining_cells:
            if cell_idx < self.num_cells:
                arrival_rates[cell_idx] = self.max_arrival_rate * 0.25  # 25% 最大流量
        
        # 应用流量强度因子
        arrival_rates = arrival_rates * self.traffic_intensity
        
        # 添加适度的随机变化（减少随机性）
        arrival_rates = arrival_rates * (0.9 + 0.2 * np.random.random(self.num_cells))
        
        # 确保在合理范围内
        arrival_rates = np.clip(arrival_rates, self.min_arrival_rate, self.max_arrival_rate)        
        return arrival_rates
    
    def _get_neighbor_cells(self, cell_idx):
        """
        获取指定小区的邻近小区（基于实际地理位置的邻接关系）
        
        Args:
            cell_idx: 小区索引（数组索引，从0开始）
            
        Returns:
            neighbors: 邻近小区索引列表
        """
        # 基于实际地理位置计算的邻接关系（转换为数组索引）
        neighbor_map = {
            0: [5, 3, 6, 2],      # 小区1的邻居：6,4,7,3 -> 索引5,3,6,2
            1: [6, 2, 7, 0],      # 小区2的邻居：7,3,8,1 -> 索引6,2,7,0
            2: [3, 1, 0, 9],      # 小区3的邻居：4,2,1,10 -> 索引3,1,0,9
            3: [2, 4, 11, 0],     # 小区4的邻居：3,5,12,1 -> 索引2,4,11,0
            4: [5, 3, 13, 0],     # 小区5的邻居：6,4,14,1 -> 索引5,3,13,0
            5: [6, 4, 0, 15],     # 小区6的邻居：7,5,1,16 -> 索引6,4,0,15
            6: [5, 1, 0, 17],     # 小区7的邻居：6,2,1,18 -> 索引5,1,0,17
            7: [1, 18, 8, 6],     # 小区8的邻居：2,19,9,7 -> 索引1,18,8,6
            8: [9, 7, 1, 2],      # 小区9的邻居：10,8,2,3 -> 索引9,7,1,2
            9: [2, 10, 8, 3],     # 小区10的邻居：3,11,9,4 -> 索引2,10,8,3
            10: [9, 11, 2, 3],    # 小区11的邻居：10,12,3,4 -> 索引9,11,2,3
            11: [3, 10, 12, 2],   # 小区12的邻居：4,11,13,3 -> 索引3,10,12,2
            12: [11, 13, 4, 3],   # 小区13的邻居：12,14,5,4 -> 索引11,13,4,3
            13: [4, 14, 12, 5],   # 小区14的邻居：5,15,13,6 -> 索引4,14,12,5
            14: [13, 15, 4, 5],   # 小区15的邻居：14,16,5,6 -> 索引13,15,4,5
            15: [5, 16, 14, 6],   # 小区16的邻居：6,17,15,7 -> 索引5,16,14,6
            16: [17, 15, 6, 5],   # 小区17的邻居：18,16,7,6 -> 索引17,15,6,5
            17: [6, 16, 18, 5],   # 小区18的邻居：7,17,19,6 -> 索引6,16,18,5
            18: [7, 17, 1, 6]     # 小区19的邻居：8,18,2,7 -> 索引7,17,1,6
        }
        
        return neighbor_map.get(cell_idx, [])
    
    def _update_arrival_rates(self):
        """
        更新到达率（模拟业务变化） - 减少随机性，保持热点分布稳定
        """
        # 减少随机波动，保持热点分布的稳定性
        variation = 0.05  # 降低波动幅度从0.1到0.05
        
        for cell_idx in range(self.num_cells):
            # 添加小幅随机变化
            change_factor = 1 + variation * (np.random.random() - 0.5)
            self.arrival_rates[cell_idx] *= change_factor
            
            # 确保在合理范围内
            self.arrival_rates[cell_idx] = np.clip(
                self.arrival_rates[cell_idx],
                self.min_arrival_rate,
                self.max_arrival_rate
            )
    
    def set_traffic_intensity(self, traffic_intensity: float):
        """
        设置流量强度并重新初始化到达率
        
        Args:
            traffic_intensity: 新的流量强度因子 (0.0-1.0)
        """
        self.traffic_intensity = np.clip(traffic_intensity, 0.0, 1.0)
        self.arrival_rates = self._initialize_arrival_rates()
    
    def get_performance_metrics(self):
        """
        获取性能指标
        
        Returns:
            metrics: 性能指标字典
        """
        return {
            'total_arrivals': self.total_arrivals,
            'total_served': self.total_served,
            'total_dropped': self.total_dropped,
            'avg_delay': self.total_delay / self.total_served if self.total_served > 0 else 0
        }
    
    def get_hotspot_info(self):
        """
        获取热点信息，用于调试和可视化
        
        Returns:
            hotspot_info: 热点信息字典
        """
        return {
            'arrival_rates': self.arrival_rates.copy(),
            'queue_lengths': self.get_queue_lengths(),
            'hotspot_centers': [(6, 8, 9), (1, 2, 3), (15, 16, 17)]
        }