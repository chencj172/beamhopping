"""
基线算法实现
包含贪心算法、随机算法等基线方法
"""

import numpy as np
from typing import Dict, List, Tuple


class GreedyAlgorithm:
    """
    贪心算法：优先为流量需求最大的小区分配波束和功率
    """
    
    def __init__(self):
        self.name = "Greedy"
    
    def get_action(self, observation: np.ndarray, env) -> Dict:
        """
        根据观测获取贪心动作
        
        Args:
            observation: 环境观测
            env: 环境实例
            
        Returns:
            action: 动作字典
        """
        # 从观测中提取信息
        if hasattr(env, 'channel_gains') and hasattr(env, 'traffic_requests'):
            channel_gains = env.channel_gains
            traffic_requests = env.traffic_requests
        else:
            # 从观测中解析（适应不同的观测格式）
            try:
                obs_flat = observation.flatten()
                channel_size = env.num_beams * env.num_cells
                channel_gains = obs_flat[:channel_size].reshape(env.num_beams, env.num_cells)
                traffic_requests = obs_flat[channel_size:channel_size + env.num_cells]
            except:
                # 如果解析失败，使用随机策略
                return self._random_action(env)
        
        return self.greedy_allocation(traffic_requests, channel_gains, env)
    
    def greedy_allocation(self, traffic_requests: np.ndarray, channel_gains: np.ndarray, env) -> Dict:
        """
        贪心分配算法
        
        Args:
            traffic_requests: 流量需求数组
            channel_gains: 信道增益矩阵
            env: 环境实例
            
        Returns:
            action: 动作字典
        """
        num_beams = env.num_beams
        num_cells = env.num_cells
        
        # 计算每个小区的优先级（流量需求 * 最佳信道增益）
        priorities = np.zeros(num_cells)
        best_beam_for_cell = np.zeros(num_cells, dtype=int)
        
        for cell_idx in range(num_cells):
            # 找到该小区的最佳波束
            best_beam = np.argmax(channel_gains[:, cell_idx])
            best_beam_for_cell[cell_idx] = best_beam
            
            # 计算优先级
            priorities[cell_idx] = traffic_requests[cell_idx] * channel_gains[best_beam, cell_idx]
        
        # 按优先级排序小区
        sorted_cells = np.argsort(priorities)[::-1]  # 降序排列
        
        # 分配波束
        beam_allocation = np.zeros(num_beams, dtype=int)
        used_beams = set()
        
        for cell_idx in sorted_cells:
            if len(used_beams) >= num_beams:
                break
                
            best_beam = best_beam_for_cell[cell_idx]
            
            # 如果最佳波束已被使用，寻找下一个最佳可用波束
            if best_beam in used_beams:
                available_gains = channel_gains[:, cell_idx].copy()
                for used_beam in used_beams:
                    available_gains[used_beam] = -1  # 标记为不可用
                
                if np.max(available_gains) > 0:
                    best_beam = np.argmax(available_gains)
                else:
                    continue  # 没有可用波束
            
            beam_allocation[best_beam] = cell_idx
            used_beams.add(best_beam)
        
        # 为未分配的波束分配小区（选择剩余需求最大的小区）
        for beam_idx in range(num_beams):
            if beam_idx not in used_beams:
                # 找到未被分配且流量需求最大的小区
                remaining_cells = set(range(num_cells)) - set(beam_allocation)
                if remaining_cells:
                    remaining_demands = [(cell_idx, traffic_requests[cell_idx]) 
                                       for cell_idx in remaining_cells]
                    remaining_demands.sort(key=lambda x: x[1], reverse=True)
                    beam_allocation[beam_idx] = remaining_demands[0][0]
        
        # 功率分配：根据流量需求和信道质量分配功率
        power_allocation = self._allocate_power(beam_allocation, traffic_requests, channel_gains)
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }
    
    def _allocate_power(self, beam_allocation: np.ndarray, traffic_requests: np.ndarray, 
                       channel_gains: np.ndarray) -> np.ndarray:
        """
        根据流量需求和信道质量分配功率
        """
        num_beams = len(beam_allocation)
        power_allocation = np.zeros(num_beams)
        
        # 计算每个波束的功率需求权重
        for beam_idx in range(num_beams):
            cell_idx = beam_allocation[beam_idx]
            channel_gain = channel_gains[beam_idx, cell_idx]
            traffic_demand = traffic_requests[cell_idx]
            
            # 功率权重 = 流量需求 / 信道增益（信道质量差需要更多功率）
            if channel_gain > 0:
                power_allocation[beam_idx] = traffic_demand / channel_gain
            else:
                power_allocation[beam_idx] = traffic_demand
        
        # 归一化功率分配
        total_power = np.sum(power_allocation)
        if total_power > 0:
            power_allocation = power_allocation / total_power
        else:
            power_allocation = np.ones(num_beams) / num_beams
        
        return power_allocation
    
    def _random_action(self, env) -> Dict:
        """随机动作（作为备用）"""
        beam_allocation = np.random.randint(0, env.num_cells, size=env.num_beams)
        power_allocation = np.random.random(env.num_beams)
        power_allocation = power_allocation / np.sum(power_allocation)
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }


class RandomAlgorithm:
    """
    随机算法：随机分配波束和功率
    """
    
    def __init__(self, num_beams: int, num_cells: int):
        self.name = "Random"
        self.num_beams = num_beams
        self.num_cells = num_cells
    
    def get_action(self, observation: np.ndarray, env) -> Dict:
        """
        随机生成动作
        
        Args:
            observation: 环境观测（未使用）
            env: 环境实例
            
        Returns:
            action: 随机动作字典
        """
        # 随机波束分配
        beam_allocation = np.random.randint(0, self.num_cells, size=self.num_beams)
        
        # 随机功率分配（归一化）
        power_allocation = np.random.random(self.num_beams)
        power_allocation = power_allocation / np.sum(power_allocation)
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }


class MaxChannelGainAlgorithm:
    """
    最大信道增益算法：始终选择信道增益最大的分配
    """
    
    def __init__(self):
        self.name = "MaxChannelGain"
    
    def get_action(self, observation: np.ndarray, env) -> Dict:
        """
        基于最大信道增益的动作选择
        """
        # 从观测中提取信道增益
        if hasattr(env, 'channel_gains'):
            channel_gains = env.channel_gains
        else:
            try:
                obs_flat = observation.flatten()
                channel_size = env.num_beams * env.num_cells
                channel_gains = obs_flat[:channel_size].reshape(env.num_beams, env.num_cells)
            except:
                return self._random_action(env)
        
        return self.max_gain_allocation(channel_gains, env)
    
    def max_gain_allocation(self, channel_gains: np.ndarray, env) -> Dict:
        """
        基于最大信道增益的分配
        """
        num_beams = env.num_beams
        num_cells = env.num_cells
        
        # 为每个波束选择信道增益最大的小区
        beam_allocation = np.argmax(channel_gains, axis=1)
        
        # 均匀功率分配（也可以根据信道增益加权）
        power_allocation = np.ones(num_beams) / num_beams
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }
    
    def _random_action(self, env) -> Dict:
        """随机动作（作为备用）"""
        beam_allocation = np.random.randint(0, env.num_cells, size=env.num_beams)
        power_allocation = np.ones(env.num_beams) / env.num_beams
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }


class ProportionalFairAlgorithm:
    """
    比例公平算法：平衡吞吐量和公平性
    """
    
    def __init__(self):
        self.name = "ProportionalFair"
        self.historical_rates = None
        self.alpha = 0.9  # 历史平均系数
    
    def get_action(self, observation: np.ndarray, env) -> Dict:
        """
        基于比例公平的动作选择
        """
        # 从观测中提取信息
        if hasattr(env, 'channel_gains') and hasattr(env, 'traffic_requests'):
            channel_gains = env.channel_gains
            traffic_requests = env.traffic_requests
        else:
            try:
                obs_flat = observation.flatten()
                channel_size = env.num_beams * env.num_cells
                channel_gains = obs_flat[:channel_size].reshape(env.num_beams, env.num_cells)
                traffic_requests = obs_flat[channel_size:channel_size + env.num_cells]
            except:
                return self._random_action(env)
        
        return self.proportional_fair_allocation(traffic_requests, channel_gains, env)
    
    def proportional_fair_allocation(self, traffic_requests: np.ndarray, 
                                   channel_gains: np.ndarray, env) -> Dict:
        """
        比例公平分配
        """
        num_beams = env.num_beams
        num_cells = env.num_cells
        
        # 初始化历史速率
        if self.historical_rates is None:
            self.historical_rates = np.ones(num_cells)
        
        # 计算即时速率估计
        instant_rates = np.zeros(num_cells)
        for cell_idx in range(num_cells):
            max_gain = np.max(channel_gains[:, cell_idx])
            instant_rates[cell_idx] = max_gain  # 简化的速率估计
        
        # 计算比例公平度量
        pf_metrics = np.zeros(num_cells)
        for cell_idx in range(num_cells):
            if self.historical_rates[cell_idx] > 0:
                pf_metrics[cell_idx] = instant_rates[cell_idx] / self.historical_rates[cell_idx]
            else:
                pf_metrics[cell_idx] = instant_rates[cell_idx]
            
            # 考虑流量需求
            pf_metrics[cell_idx] *= traffic_requests[cell_idx]
        
        # 贪心分配：选择PF度量最高的小区
        sorted_cells = np.argsort(pf_metrics)[::-1]
        
        beam_allocation = np.zeros(num_beams, dtype=int)
        used_beams = set()
        
        for cell_idx in sorted_cells:
            if len(used_beams) >= num_beams:
                break
            
            # 为该小区选择最佳可用波束
            best_beam = -1
            best_gain = -1
            
            for beam_idx in range(num_beams):
                if beam_idx not in used_beams:
                    gain = channel_gains[beam_idx, cell_idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_beam = beam_idx
            
            if best_beam >= 0:
                beam_allocation[best_beam] = cell_idx
                used_beams.add(best_beam)
        
        # 为未分配的波束随机分配
        for beam_idx in range(num_beams):
            if beam_idx not in used_beams:
                remaining_cells = set(range(num_cells)) - set(beam_allocation)
                if remaining_cells:
                    beam_allocation[beam_idx] = np.random.choice(list(remaining_cells))
                else:
                    beam_allocation[beam_idx] = np.random.randint(0, num_cells)
        
        # 功率分配
        power_allocation = self._allocate_power_pf(beam_allocation, pf_metrics, channel_gains)
        
        # 更新历史速率
        self._update_historical_rates(instant_rates)
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }
    
    def _allocate_power_pf(self, beam_allocation: np.ndarray, pf_metrics: np.ndarray,
                          channel_gains: np.ndarray) -> np.ndarray:
        """基于比例公平度量的功率分配"""
        num_beams = len(beam_allocation)
        power_allocation = np.zeros(num_beams)
        
        for beam_idx in range(num_beams):
            cell_idx = beam_allocation[beam_idx]
            power_allocation[beam_idx] = pf_metrics[cell_idx]
        
        # 归一化
        total_power = np.sum(power_allocation)
        if total_power > 0:
            power_allocation = power_allocation / total_power
        else:
            power_allocation = np.ones(num_beams) / num_beams
        
        return power_allocation
    
    def _update_historical_rates(self, instant_rates: np.ndarray):
        """更新历史平均速率"""
        self.historical_rates = (self.alpha * self.historical_rates + 
                               (1 - self.alpha) * instant_rates)
    
    def _random_action(self, env) -> Dict:
        """随机动作（作为备用）"""
        beam_allocation = np.random.randint(0, env.num_cells, size=env.num_beams)
        power_allocation = np.ones(env.num_beams) / env.num_beams
        
        return {
            'beam_allocation': beam_allocation,
            'power_allocation': power_allocation
        }


# 为了兼容性，创建算法字典
ALGORITHMS = {
    'greedy': GreedyAlgorithm,
    'random': RandomAlgorithm,
    'max_gain': MaxChannelGainAlgorithm,
    'proportional_fair': ProportionalFairAlgorithm
}


def get_algorithm(name: str, **kwargs):
    """
    获取指定名称的算法实例
    
    Args:
        name: 算法名称
        **kwargs: 算法初始化参数
        
    Returns:
        algorithm: 算法实例
    """
    if name.lower() in ALGORITHMS:
        return ALGORITHMS[name.lower()](**kwargs)
    else:
        raise ValueError(f"未知算法: {name}")
