import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple, List, Optional

from rl.models import DiscreteActor, ContinuousActor, Critic
from rl.buffer import RolloutBuffer


class PPO:
    """
    PPO算法实现
    
    包含两个策略网络：
    1. 离散策略网络：用于波束分配（离散动作空间）
    2. 连续策略网络：用于功率分配（连续动作空间）
    """
    
    def __init__(self,
                 state_dim: int,
                 discrete_action_dim: int,
                 continuous_action_dim: int,
                 discrete_action_space_size: int,
                 lr_actor: float = 0.0003,
                 lr_critic: float = 0.001,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 update_epochs: int = 10,
                 mini_batch_size: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化PPO算法
        
        Args:
            state_dim: 状态维度
            discrete_action_dim: 离散动作维度（波束数量）
            continuous_action_dim: 连续动作维度（波束数量）
            discrete_action_space_size: 离散动作空间大小（小区数量）
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_ratio: PPO裁剪比例
            value_clip_ratio: 价值函数裁剪比例
            entropy_coef: 熵正则化系数
            update_epochs: 每次更新的epoch数
            mini_batch_size: 小批量大小
            device: 计算设备
        """
        self.state_dim = state_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_space_size = discrete_action_space_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        
        # 创建离散策略网络（波束分配）
        self.discrete_actor = DiscreteActor(
            state_dim=state_dim,
            action_dim=discrete_action_dim,
            action_space_size=discrete_action_space_size
        ).to(device)
        
        # 创建连续策略网络（功率分配）
        self.continuous_actor = ContinuousActor(
            state_dim=state_dim,
            action_dim=continuous_action_dim
        ).to(device)
        
        # 创建价值网络
        self.critic = Critic(state_dim=state_dim).to(device)
        
        # 创建优化器
        self.discrete_actor_optimizer = optim.Adam(self.discrete_actor.parameters(), lr=lr_actor)
        self.continuous_actor_optimizer = optim.Adam(self.continuous_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 创建经验回放缓冲区
        self.buffer = RolloutBuffer()
    
    def select_action(self, state):
        """
        选择动作
        
        Args:
            state: 环境状态
        
        Returns:
            action: 动作字典，包含波束分配和功率分配
            discrete_action_log_prob: 离散动作对数概率
            continuous_action_log_prob: 连续动作对数概率
            state_value: 状态价值
        """
        # 转换为张量
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # 如果状态是一维的，添加批次维度
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # 获取离散动作分布（波束分配）
        discrete_action_probs = self.discrete_actor(state_tensor)  # shape: [batch_size, action_dim, action_space_size]
        
        # 创建多个分类分布（每个波束一个）
        discrete_action = []
        discrete_action_log_prob_list = []
        
        # 对每个波束采样动作
        for i in range(self.discrete_action_dim):
            dist = Categorical(probs=discrete_action_probs[0, i])
            action = dist.sample()
            discrete_action.append(action.item())
            discrete_action_log_prob_list.append(dist.log_prob(action))
        
        # 将离散动作对数概率求和
        discrete_action_log_prob = torch.stack(discrete_action_log_prob_list).sum().item()
        
        # 获取连续动作分布（功率分配）
        continuous_action_mean, continuous_action_std = self.continuous_actor(state_tensor)
        continuous_dist = Normal(continuous_action_mean, continuous_action_std)
        
        # 采样连续动作
        # continuous_action = continuous_dist.sample()
        continuous_action = torch.sigmoid(continuous_dist.sample())
        # print("1、", continuous_action)
        continuous_action_log_prob_tensor = continuous_dist.log_prob(continuous_action)
        
        # 计算连续动作对数概率（求和）
        continuous_action_log_prob = continuous_action_log_prob_tensor.sum().item()
        
        # 获取状态价值
        state_value = self.critic(state_tensor)
        
        # 转换为numpy数组
        continuous_action = continuous_action.detach().cpu().numpy()
        
        # print("2、", continuous_action)
        state_value = state_value.detach().cpu().item()
        
        # 创建动作字典
        action = {
            'beam_allocation': discrete_action,
            'power_allocation': continuous_action
        }
        
        return action, discrete_action_log_prob, continuous_action_log_prob, state_value
    
    def evaluate_actions(self, states, discrete_actions, continuous_actions):
        """
        评估动作
        
        Args:
            states: 状态批量
            discrete_actions: 离散动作批量
            continuous_actions: 连续动作批量
        
        Returns:
            discrete_action_log_probs: 离散动作对数概率
            continuous_action_log_probs: 连续动作对数概率
            state_values: 状态价值
            entropy: 熵
        """
        # 获取离散动作分布（波束分配）
        discrete_action_probs = self.discrete_actor(states)  # shape: [batch_size, action_dim, action_space_size]
        
        # 创建多个分类分布（每个波束一个）
        batch_size = states.size(0)
        discrete_action_log_probs_list = []
        entropy_discrete = 0
        
        # 对每个批次样本和每个波束计算对数概率
        for b in range(batch_size):
            batch_log_probs = []
            for i in range(self.discrete_action_dim):
                dist = Categorical(probs=discrete_action_probs[b, i])
                action = discrete_actions[b, i]
                batch_log_probs.append(dist.log_prob(action))
                entropy_discrete += dist.entropy()
            # 对每个批次，将所有波束的对数概率求和
            discrete_action_log_probs_list.append(torch.sum(torch.stack(batch_log_probs)))
        
        # 将所有批次的对数概率堆叠起来
        discrete_action_log_probs = torch.stack(discrete_action_log_probs_list)
        entropy_discrete = entropy_discrete / (batch_size * self.discrete_action_dim)  # 计算平均熵
        
        # 获取连续动作分布（功率分配）
        continuous_action_mean, continuous_action_std = self.continuous_actor(states)
        continuous_dist = Normal(continuous_action_mean, continuous_action_std)
        
        # 计算连续动作对数概率
        continuous_action_log_probs_raw = continuous_dist.log_prob(continuous_actions)
        # print(f"continuous_action_log_probs_raw shape: {continuous_action_log_probs_raw.shape}")
        # print(f"continuous_actions shape: {continuous_actions.shape}")
        
        # 对每个批次，将所有连续动作的对数概率求和
        continuous_action_log_probs = continuous_action_log_probs_raw.sum(dim=1)
        # print(f"continuous_action_log_probs shape after sum: {continuous_action_log_probs.shape}")
        
        # 计算连续动作的熵
        entropy_continuous = continuous_dist.entropy().mean()
        
        # 计算总熵（离散熵已在上面计算）
        entropy = entropy_discrete + entropy_continuous
        
        # 获取状态价值
        state_values = self.critic(states)
        
        return discrete_action_log_probs, continuous_action_log_probs, state_values, entropy
    
    def update(self):
        """
        更新策略网络和价值网络
        
        Returns:
            loss_info: 损失信息字典
        """
        # 计算优势函数和回报
        self._compute_advantages_and_returns()
        
        # 获取数据
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        discrete_actions = torch.LongTensor(np.array(self.buffer.discrete_actions)).to(self.device)
        continuous_actions = torch.FloatTensor(np.array(self.buffer.continuous_actions)).to(self.device)
        old_discrete_action_log_probs = torch.FloatTensor(np.array(self.buffer.discrete_action_log_probs)).to(self.device)
        old_continuous_action_log_probs = torch.FloatTensor(np.array(self.buffer.continuous_action_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(self.buffer.returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(self.buffer.advantages)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.buffer.values)).to(self.device)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 记录损失信息
        loss_info = {
            'actor_loss': 0,
            'critic_loss': 0,
            'entropy': 0
        }
        
        # 多次更新
        for _ in range(self.update_epochs):
            # 生成小批量索引
            indices = np.random.permutation(len(states))
            
            # 按小批量更新
            for start_idx in range(0, len(states), self.mini_batch_size):
                # 获取小批量索引
                batch_indices = indices[start_idx:start_idx + self.mini_batch_size]
                
                # 获取小批量数据
                batch_states = states[batch_indices]
                batch_discrete_actions = discrete_actions[batch_indices]
                batch_continuous_actions = continuous_actions[batch_indices]
                batch_old_discrete_action_log_probs = old_discrete_action_log_probs[batch_indices]
                batch_old_continuous_action_log_probs = old_continuous_action_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # 评估动作
                new_discrete_action_log_probs, new_continuous_action_log_probs, new_values, entropy = \
                    self.evaluate_actions(batch_states, batch_discrete_actions, batch_continuous_actions)
                
                # 打印形状信息，用于调试
                # print(f"new_continuous_action_log_probs shape: {new_continuous_action_log_probs.shape}")
                # print(f"batch_old_continuous_action_log_probs shape: {batch_old_continuous_action_log_probs.shape}")
                
                # 确保连续动作对数概率的形状正确
                if new_continuous_action_log_probs.shape != batch_old_continuous_action_log_probs.shape:
                    # 如果形状不匹配，可能是因为batch_old_continuous_action_log_probs是标量值
                    # 而new_continuous_action_log_probs是每个动作维度的对数概率
                    # 我们需要将new_continuous_action_log_probs求和，使其与buffer中存储的格式一致
                    if len(new_continuous_action_log_probs.shape) > 1:
                        new_continuous_action_log_probs = new_continuous_action_log_probs.sum(dim=1)
                        # print(f"After sum, new_continuous_action_log_probs shape: {new_continuous_action_log_probs.shape}")
                
                # 计算离散动作比率（现在是一维张量）
                discrete_ratio = torch.exp(new_discrete_action_log_probs - batch_old_discrete_action_log_probs)
                
                # 计算连续动作比率
                # 由于形状不匹配，我们需要调整形状
                # 假设new_continuous_action_log_probs的形状是[batch_size]，batch_old_continuous_action_log_probs的形状是[batch_size]
                # 我们需要确保两个张量的形状匹配
                # print(f"new_continuous_action_log_probs shape: {new_continuous_action_log_probs.shape}")
                # print(f"batch_old_continuous_action_log_probs shape: {batch_old_continuous_action_log_probs.shape}")
                
                # 计算连续动作比率
                continuous_ratio = torch.exp(new_continuous_action_log_probs - batch_old_continuous_action_log_probs)
                
                # 计算总比率（乘积）
                ratio = discrete_ratio * continuous_ratio
                
                # 计算策略损失（PPO-Clip）
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失（带裁剪）
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.value_clip_ratio,
                    self.value_clip_ratio
                )
                value_loss1 = (new_values - batch_returns).pow(2)
                value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # 计算总损失
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # 计算梯度并更新所有网络
                # 首先清空所有梯度
                self.discrete_actor_optimizer.zero_grad()
                self.continuous_actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # 计算actor损失的梯度
                actor_loss.backward(retain_graph=True)
                
                # 计算critic损失的梯度
                critic_loss.backward()
                
                # 裁剪梯度并更新网络
                nn.utils.clip_grad_norm_(self.discrete_actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.continuous_actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                
                # 更新网络参数
                self.discrete_actor_optimizer.step()
                self.continuous_actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 更新损失信息
                loss_info['actor_loss'] += actor_loss.item()
                loss_info['critic_loss'] += critic_loss.item()
                loss_info['entropy'] += entropy.item()
        
        # 计算平均损失
        num_updates = self.update_epochs * (len(states) // self.mini_batch_size + int(len(states) % self.mini_batch_size > 0))
        loss_info['actor_loss'] /= num_updates
        loss_info['critic_loss'] /= num_updates
        loss_info['entropy'] /= num_updates
        
        # 清空缓冲区
        self.buffer.clear()
        
        return loss_info
    
    def _compute_advantages_and_returns(self):
        """
        计算优势函数和回报
        """
        # 获取数据
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)
        
        # 计算GAE和回报
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae_lam = 0
        
        # 反向遍历
        for t in reversed(range(len(rewards))):
            # 计算时序差分目标
            if t == len(rewards) - 1:
                # 最后一步
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # 计算GAE
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
            
            # 计算回报
            returns[t] = advantages[t] + values[t]
        
        # 存储到缓冲区
        self.buffer.advantages = advantages
        self.buffer.returns = returns
    
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'discrete_actor': self.discrete_actor.state_dict(),
            'continuous_actor': self.continuous_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'discrete_actor_optimizer': self.discrete_actor_optimizer.state_dict(),
            'continuous_actor_optimizer': self.continuous_actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        # 确保模型被加载到正确的设备上
        checkpoint = torch.load(path, map_location=self.device)
        
        self.discrete_actor.load_state_dict(checkpoint['discrete_actor'])
        self.continuous_actor.load_state_dict(checkpoint['continuous_actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        
        self.discrete_actor_optimizer.load_state_dict(checkpoint['discrete_actor_optimizer'])
        self.continuous_actor_optimizer.load_state_dict(checkpoint['continuous_actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])