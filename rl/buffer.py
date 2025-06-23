import numpy as np
from typing import List, Dict, Any


class RolloutBuffer:
    """
    经验回放缓冲区
    
    用于存储训练数据
    """
    
    def __init__(self):
        """
        初始化经验回放缓冲区
        """
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_action_log_probs = []
        self.continuous_action_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self,
            state,
            discrete_action,
            continuous_action,
            discrete_action_log_prob,
            continuous_action_log_prob,
            reward,
            value,
            done):
        """
        添加一条经验
        
        Args:
            state: 状态
            discrete_action: 离散动作
            continuous_action: 连续动作
            discrete_action_log_prob: 离散动作对数概率
            continuous_action_log_prob: 连续动作对数概率
            reward: 奖励
            value: 状态价值
            done: 是否结束
        """
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.discrete_action_log_probs.append(discrete_action_log_prob)
        self.continuous_action_log_probs.append(continuous_action_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """
        清空缓冲区
        """
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_action_log_probs = []
        self.continuous_action_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        """
        获取缓冲区大小
        
        Returns:
            size: 缓冲区大小
        """
        return len(self.states)