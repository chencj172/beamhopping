import numpy as np
from typing import Dict, List, Tuple


class Metrics:
    """
    性能评估指标类
    
    用于计算和评估系统性能
    """
    
    @staticmethod
    def calculate_throughput(served_packets: np.ndarray, packet_size_kb: float = 100.0):
        """
        计算吞吐量
        
        Args:
            served_packets: 服务的数据包数量数组，shape=(num_cells,)
            packet_size_kb: 数据包大小 (KB)
        
        Returns:
            throughput_mbps: 吞吐量 (Mbps)
        """
        # 计算总吞吐量 (Mbps)
        # 假设每个时隙为1秒
        throughput_mbps = np.sum(served_packets) * packet_size_kb * 8 / 1000
        
        return throughput_mbps
    
    @staticmethod
    def calculate_average_delay(delays: List[float]):
        """
        计算平均延迟
        
        Args:
            delays: 延迟列表，单位：时隙
        
        Returns:
            avg_delay: 平均延迟，单位：时隙
        """
        # 计算平均延迟
        avg_delay = np.mean(delays) if delays else 0
        
        return avg_delay
    
    @staticmethod
    def calculate_packet_loss_rate(dropped_packets: int, total_packets: int):
        """
        计算丢包率
        
        Args:
            dropped_packets: 丢弃的数据包数量
            total_packets: 总数据包数量
        
        Returns:
            loss_rate: 丢包率
        """
        # 计算丢包率
        loss_rate = dropped_packets / total_packets if total_packets > 0 else 0
        
        return loss_rate
    
    @staticmethod
    def calculate_spectral_efficiency(data_rates: np.ndarray, bandwidth_mhz: float = 200.0):
        """
        计算频谱效率
        
        Args:
            data_rates: 数据率数组，shape=(num_cells,)，单位：bits/s
            bandwidth_mhz: 带宽 (MHz)
        
        Returns:
            spectral_efficiency: 频谱效率 (bits/s/Hz)
        """
        # 计算总数据率 (bits/s)
        total_data_rate = np.sum(data_rates)
        
        
        # 计算频谱效率 (bits/s/Hz)
        spectral_efficiency = total_data_rate / (bandwidth_mhz * 1e6)
        
        return spectral_efficiency
    
    @staticmethod
    def calculate_power_efficiency(data_rates: np.ndarray, power_dbw: float = 39.0):
        """
        计算功率效率
        
        Args:
            data_rates: 数据率数组，shape=(num_cells,)，单位：bits/s
            power_dbw: 总功率 (dBW)
        
        Returns:
            power_efficiency: 功率效率 (bits/s/W)
        """
        # 计算总数据率 (bits/s)
        total_data_rate = np.sum(data_rates)
        
        # 将功率从dBW转换为W
        power_w = 10 ** (power_dbw / 10)
        
        # 计算功率效率 (bits/s/W)
        power_efficiency = total_data_rate / power_w
        
        return power_efficiency
    
    @staticmethod
    def calculate_fairness_index(data_rates: np.ndarray):
        """
        计算公平性指数（Jain's Fairness Index）
        
        Args:
            data_rates: 数据率数组，shape=(num_cells,)，单位：bits/s
        
        Returns:
            fairness_index: 公平性指数，范围：[0, 1]，值越大表示越公平
        """
        # 计算公平性指数
        # J(x) = (sum(x_i))^2 / (n * sum(x_i^2))
        n = len(data_rates)
        if n == 0 or np.sum(data_rates) == 0:
            return 0
        
        fairness_index = np.sum(data_rates) ** 2 / (n * np.sum(data_rates ** 2))
        
        return fairness_index
    
    @staticmethod
    def calculate_satisfaction_ratio(served_traffic: np.ndarray, requested_traffic: np.ndarray):
        """
        计算满足率
        
        Args:
            served_traffic: 服务的业务量数组，shape=(num_cells,)
            requested_traffic: 请求的业务量数组，shape=(num_cells,)
        
        Returns:
            satisfaction_ratio: 满足率，范围：[0, 1]
        """
        # 计算每个小区的满足率
        cell_satisfaction = np.zeros_like(served_traffic)
        for i in range(len(served_traffic)):
            if requested_traffic[i] > 0:
                cell_satisfaction[i] = min(served_traffic[i] / requested_traffic[i], 1.0)
            else:
                cell_satisfaction[i] = 1.0
        
        # 计算总体满足率
        satisfaction_ratio = np.mean(cell_satisfaction)
        
        return satisfaction_ratio
    
    @staticmethod
    def calculate_overall_performance(throughput: float, delay: float, loss_rate: float, 
                                    fairness: float, satisfaction: float,
                                    weights: Dict[str, float] = None):
        """
        计算总体性能指标
        
        Args:
            throughput: 吞吐量 (Mbps)
            delay: 延迟 (时隙)
            loss_rate: 丢包率
            fairness: 公平性指数
            satisfaction: 满足率
            weights: 权重字典，默认为None（等权重）
        
        Returns:
            overall_performance: 总体性能指标，范围：[0, 1]，值越大表示性能越好
        """
        # 默认权重
        if weights is None:
            weights = {
                'throughput': 0.3,
                'delay': 0.2,
                'loss_rate': 0.2,
                'fairness': 0.15,
                'satisfaction': 0.15
            }
        
        # 归一化吞吐量（假设最大吞吐量为1000 Mbps）
        norm_throughput = min(throughput / 1000, 1.0)
        
        # 归一化延迟（假设最大延迟为10时隙）
        norm_delay = max(1.0 - delay / 10, 0.0)
        
        # 归一化丢包率
        norm_loss_rate = 1.0 - loss_rate
        
        # 计算总体性能指标
        overall_performance = (
            weights['throughput'] * norm_throughput +
            weights['delay'] * norm_delay +
            weights['loss_rate'] * norm_loss_rate +
            weights['fairness'] * fairness +
            weights['satisfaction'] * satisfaction
        )
        
        return overall_performance