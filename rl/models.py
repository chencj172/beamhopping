import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DiscreteActor(nn.Module):
    """
    离散Actor网络
    
    用于波束分配（离散动作空间）
    """
    
    def __init__(self, state_dim: int, action_dim: int, action_space_size: int, hidden_dim: int = 256):
        """
        初始化离散Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（波束数量）
            action_space_size: 动作空间大小（小区数量）
            hidden_dim: 隐藏层维度
        """
        super(DiscreteActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_size = action_space_size
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 为每个波束创建一个输出头
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_space_size) for _ in range(action_dim)
        ])
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量
        
        Returns:
            action_probs: 动作概率，shape=(batch_size, action_dim, action_space_size)
        """
        # 检查输入状态的维度，如果是1D，则添加批次维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加批次维度
            
        # 提取特征
        features = self.feature_extractor(state)
        
        # 计算每个波束的动作概率
        action_logits = [head(features) for head in self.action_heads]
        
        # 将动作概率堆叠起来
        action_logits = torch.stack(action_logits, dim=1)
        
        # 检查logits是否包含NaN或Inf
        if torch.any(torch.isnan(action_logits)) or torch.any(torch.isinf(action_logits)):
            print(f"Warning: NaN/Inf detected in action_logits: {action_logits}")
            # 用小的随机值替换NaN/Inf
            action_logits = torch.where(
                torch.isnan(action_logits) | torch.isinf(action_logits),
                torch.randn_like(action_logits) * 0.01,
                action_logits
            )
        
        # 数值稳定的softmax：先减去最大值
        action_logits_stable = action_logits - torch.max(action_logits, dim=-1, keepdim=True)[0]
        
        # 应用softmax获取概率分布
        action_probs = F.softmax(action_logits_stable, dim=-1)
        
        # 检查概率是否包含NaN
        if torch.any(torch.isnan(action_probs)):
            print(f"Warning: NaN detected in action_probs after softmax")
            # 设置为均匀分布
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        
        # 确保概率总和为1（数值稳定性）
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return action_probs


class ContinuousActor(nn.Module):
    """
    连续Actor网络
    
    用于功率分配（连续动作空间）
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化连续Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（波束数量）
            hidden_dim: 隐藏层维度
        """
        super(ContinuousActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值输出层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # 标准差输出层
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
        
        Returns:
            mean: 动作均值
            std: 动作标准差
        """
        # 检查输入状态的维度，如果是1D，则添加批次维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加批次维度
            
        # 提取特征
        features = self.feature_extractor(state)
        
        # 检查特征是否包含NaN
        if torch.any(torch.isnan(features)):
            print(f"Warning: NaN detected in continuous actor features")
            print(f"State shape: {state.shape}, State mean: {state.mean()}, State std: {state.std()}")
            # 用小的随机值替换NaN
            features = torch.where(
                torch.isnan(features),
                torch.randn_like(features) * 0.01,
                features
            )
        
        # 计算均值（使用tanh而不是sigmoid以避免饱和）
        mean_logits = self.mean_layer(features)
        # 检查均值logits是否包含NaN
        if torch.any(torch.isnan(mean_logits)):
            print(f"Warning: NaN detected in mean_logits")
            mean_logits = torch.where(
                torch.isnan(mean_logits),
                torch.zeros_like(mean_logits),
                mean_logits
            )
        
        # 使用tanh激活并缩放到[0,1]范围，避免sigmoid的数值问题
        mean = (torch.tanh(mean_logits) + 1.0) / 2.0  # 将[-1,1]映射到[0,1]
        
        # 计算标准差
        log_std = self.log_std_layer(features)
        # 检查log_std是否包含NaN
        if torch.any(torch.isnan(log_std)):
            print(f"Warning: NaN detected in log_std")
            log_std = torch.where(
                torch.isnan(log_std),
                torch.ones_like(log_std) * (-2.0),  # 默认较小的标准差
                log_std
            )
        
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        std = torch.exp(log_std)
        
        # 最终检查输出是否包含NaN
        if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
            print(f"Warning: NaN in final continuous actor output")
            print(f"Mean NaN count: {torch.isnan(mean).sum()}")
            print(f"Std NaN count: {torch.isnan(std).sum()}")
            # 设置为安全的默认值
            if torch.any(torch.isnan(mean)):
                mean = torch.where(torch.isnan(mean), torch.tensor(0.5, device=mean.device), mean)
            if torch.any(torch.isnan(std)):
                std = torch.where(torch.isnan(std), torch.tensor(0.1, device=std.device), std)
        
        return mean, std


class Critic(nn.Module):
    """
    Critic网络
    
    用于估计状态价值
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        
        # 网络结构
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量
        
        Returns:
            value: 状态价值
        """
        # 检查输入状态的维度，如果是1D，则添加批次维度
        if state.dim() == 1:
            state = state.unsqueeze(0)  # 添加批次维度
            
        # 计算状态价值
        value = self.network(state)
        
        return value