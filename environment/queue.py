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
                 max_queue_size: int = 100,
                 packet_ttl: int = 10,
                 min_arrival_rate: float = 0.1,  # 最小到达率（数据包/时隙）
                 max_arrival_rate: float = 1.0,  # 最大到达率（数据包/时隙）
                 packet_size_kb: float = 100.0):  # 数据包大小 (KB)
        """
        初始化队列模型
        
        Args:
            num_cells: 小区数量
            max_queue_size: 队列最大容量
            packet_ttl: 数据包生存时间(时隙)
            min_arrival_rate: 最小到达率（数据包/时隙）
            max_arrival_rate: 最大到达率（数据包/时隙）
            packet_size_kb: 数据包大小 (KB)
        """
        self.num_cells = num_cells
        self.max_queue_size = max_queue_size
        self.packet_ttl = packet_ttl
        self.min_arrival_rate = min_arrival_rate
        self.max_arrival_rate = max_arrival_rate
        self.packet_size_kb = packet_size_kb
        
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
    
    def update(self, data_rates):
        """
        更新队列状态
        
        Args:
            data_rates: 数据率数组，单位：bits/s
        
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
            max_serve = int(data_rates[cell_idx] / (self.packet_size_kb * 8 * 1000))
            
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
            arrival_count = np.random.poisson(self.arrival_rates[cell_idx])
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
    
    def _initialize_arrival_rates(self):
        """
        初始化到达率
        
        Returns:
            arrival_rates: 到达率数组
        """
        # 使用不同的到达率模拟不同小区的业务需求
        # 中心小区通常业务量较大，边缘小区业务量较小
        
        # 计算每个小区到中心的距离
        center_idx = 0  # 假设中心小区索引为0
        distances = np.zeros(self.num_cells)
        
        for cell_idx in range(1, self.num_cells):
            # 距离越远，到达率越小
            distances[cell_idx] = cell_idx / self.num_cells
        
        # 根据距离计算到达率
        arrival_rates = self.max_arrival_rate - distances * (self.max_arrival_rate - self.min_arrival_rate)
        
        # 添加随机变化
        arrival_rates = arrival_rates * (0.8 + 0.4 * np.random.random(self.num_cells))
        
        return arrival_rates
    
    def _update_arrival_rates(self):
        """
        更新到达率（模拟业务变化）
        """
        # 添加随机波动
        variation = 0.1  # 波动幅度
        
        for cell_idx in range(self.num_cells):
            # 添加随机变化
            self.arrival_rates[cell_idx] *= (1 + variation * (np.random.random() - 0.5))
            
            # 确保在合理范围内
            self.arrival_rates[cell_idx] = np.clip(
                self.arrival_rates[cell_idx],
                self.min_arrival_rate,
                self.max_arrival_rate
            )
    
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