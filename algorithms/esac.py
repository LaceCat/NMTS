"""
ESAC (Ensemble Soft Actor-Critic) 算法实现

基于: Jia et al., "Ensemble reinforcement learning for optimizing the energy efficiency
      index in the thickening-dewatering process", Computers in Industry 175 (2026) 104431

核心设计 (参照论文 Algorithm 1 + Fig. 3):
1. n 个 Actor 并行探索，每个独立参数
2. 每个 Actor 生成动作，选 reward/Q 最高的 action 存入 replay buffer
3. 用最优 action 更新所有 Actor 和共享 Critic
4. 双 Critic 取 min (防止 Q 值高估)
5. 熵正则化 (SAC 最大熵框架)

论文超参数 (Table 1):
    优化器: Adam
    折扣因子 γ: 0.99
    经验回放: 4000
    批次大小: 128
    Actor LR: 3e-4
    Critic LR: 3e-3
    隐藏层: 256
    激活函数: ReLU
    熵系数 α: 0.99
    ESAC actors: 5
    Polyak: 0.997
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

from utils.replay_buffer import PrioritizedReplayBuffer


# 论文 F Table 1 超参数
STATE_DIM = 9
ACTION_DIM = 2
HIDDEN_DIM = 256
BUFFER_CAPACITY = 4000        # 论文: 4000
BATCH_SIZE = 128              # 论文: 128
GAMMA = 0.99                  # 论文: 0.99
TAU = 0.003                   # 论文 Polyak=0.997 → τ = 1 - 0.997 = 0.003
LR_ACTOR = 3e-4               # 论文: 3e-4
LR_CRITIC = 3e-3              # 论文: 3e-3
ALPHA = 0.99                  # 论文熵系数: 0.99
NUM_ACTORS = 5                # 论文 ESAC actors: 5
EXPLORATION_NOISE = 0.1       # 论文 TD3 act noise: 0.1
WARMUP_STEPS = 1000
EPOCHS = 1000
EPISODES_PER_EPOCH = 5
CHECKPOINT_INTERVAL = 100
ACTION_BOUNDS_LOW = np.array([0.0, 0.0], dtype=np.float32)
ACTION_BOUNDS_HIGH = np.array([50.0, 70.0], dtype=np.float32)

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class SacActor(nn.Module):
    """
    SAC Actor 网络 - 输出动作分布 (均值 + 对数标准差)

    论文结构: 256 隐藏层, ReLU 激活 (Table 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False):
        """采样动作 (带 reparameterization trick)"""
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(mean[..., 0])
            return action, log_prob

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization
        action = torch.tanh(z)

        # 计算对数概率 (含 tanh 修正)
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob


class SharedCritic(nn.Module):
    """
    共享 Critic 网络 (所有 Actor 共用)

    论文结构: 256 隐藏层, ReLU 激活 (Table 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ESACAgent:
    """
    ESAC 智能体 (Ensemble Soft Actor-Critic)

    参照论文 Algorithm 1:
    - n 个 Actor 并行生成动作
    - 选 Q 值最高的动作执行
    - 用该动作更新所有 Actor 和 Critic
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
        buffer_capacity: int = BUFFER_CAPACITY,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        tau: float = TAU,
        lr_actor: float = LR_ACTOR,
        lr_critic: float = LR_CRITIC,
        alpha: float = ALPHA,
        num_actors: int = NUM_ACTORS,
        device: str = 'cpu',
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.num_actors = num_actors
        self.batch_size = batch_size
        self.device = device
        self.update_step = 0

        # 经验回放 (论文使用标准 replay，这里保留 prioritized)
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

        # n 个 Actor (每个独立参数，随机初始化)
        self.actors = [
            SacActor(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_actors)
        ]
        self.actor_targets = [
            SacActor(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_actors)
        ]
        for i in range(num_actors):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())

        # 共享双 Critic (论文 twin Q 网络)
        self.critic1 = SharedCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = SharedCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = SharedCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = SharedCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 冻结目标网络
        for net in self.actor_targets + [self.critic1_target, self.critic2_target]:
            for param in net.parameters():
                param.requires_grad = False

        # 优化器 (论文: Adam)
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # 动作范围
        self.action_low = torch.tensor(ACTION_BOUNDS_LOW, dtype=torch.float32).to(device)
        self.action_high = torch.tensor(ACTION_BOUNDS_HIGH, dtype=torch.float32).to(device)

        # 指标
        self.all_rewards = []
        self.all_final_masses = []
        self.all_energy_costs = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作

        论文 Algorithm 1 line 8-12:
        每个 Actor 生成动作，选 Q 值最高的
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            best_action = None
            best_q = -float('inf')

            for i in range(self.num_actors):
                action_raw, _ = self.actors[i].sample(state_tensor, deterministic=deterministic)
                # 映射到实际动作范围
                action = self._scale_action_tensor(action_raw)

                # 用 Critic1 评估 Q 值
                q = self.critic1(state_tensor, action).item()

                if q > best_q:
                    best_q = q
                    best_action = action.cpu().numpy()[0]

        # 裁剪
        action = np.clip(best_action, ACTION_BOUNDS_LOW, ACTION_BOUNDS_HIGH)
        return action

    def _scale_action_tensor(self, raw_action: torch.Tensor) -> torch.Tensor:
        """将 tanh 输出 [-1, 1] 映射到实际动作范围"""
        low = self.action_low
        high = self.action_high
        return 0.5 * (high - low) * (raw_action + 1) + low

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        执行一次更新

        论文 Algorithm 1 line 15-20:
        1. 从 buffer 采样
        2. 更新 Critic (Bellman 误差)
        3. 更新所有 Actor (基于相同采样数据)
        4. 软更新目标网络
        """
        if self.buffer.size < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, self.device)
        if batch is None:
            return {}

        states, actions, rewards, next_states, dones, indices, weights = batch

        # ========== 更新 Critic ==========
        with torch.no_grad():
            # 目标动作 (用第一个 actor 的目标网络)
            next_action_raw, next_log_prob = self.actor_targets[0].sample(next_states)
            next_action = self._scale_action_tensor(next_action_raw)

            # 双 Q 目标取 min (论文 twin Q)
            target_q1 = self.critic1_target(next_states, next_action)
            target_q2 = self.critic2_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)

            # SAC 目标: r + γ * (Q - α * log_prob)
            q_target = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_log_prob.unsqueeze(1))

        # Critic 损失
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        # TD 误差 (用于 PER 优先级更新)
        td_errors1 = (q_target - q1).detach().cpu().numpy().flatten()
        td_errors2 = (q_target - q2).detach().cpu().numpy().flatten()
        td_errors = 0.5 * (np.abs(td_errors1) + np.abs(td_errors2))

        critic1_loss = (weights * F.mse_loss(q1, q_target, reduction='none')).mean()
        critic2_loss = (weights * F.mse_loss(q2, q_target, reduction='none')).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新优先级
        self.buffer.update_priorities(indices, td_errors)

        # ========== 更新所有 Actor ==========
        actor_losses = []
        for i in range(self.num_actors):
            # 采样动作
            action_raw, log_prob = self.actors[i].sample(states)
            action = self._scale_action_tensor(action_raw)

            # Actor 损失: 最大化 Q - α * log_prob (论文式 11)
            q = self.critic1(states, action)
            actor_loss = -(q - self.alpha * log_prob).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())

        # ========== 软更新目标网络 ==========
        for i in range(self.num_actors):
            self._soft_update(self.actor_targets[i], self.actors[i], self.tau)
        self._soft_update(self.critic1_target, self.critic1, self.tau)
        self._soft_update(self.critic2_target, self.critic2, self.tau)

        self.update_step += 1

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss_mean': np.mean(actor_losses),
            'actor_loss_std': np.std(actor_losses),
        }

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        """软更新目标网络 (论文 Polyak=0.997 → τ=0.003)"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )

    def save(self, path: str):
        """保存模型 (所有 Actor + Critic)"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        save_dict = {
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'update_step': self.update_step,
            'num_actors': self.num_actors,
        }
        for i in range(self.num_actors):
            save_dict[f'actor_{i}'] = self.actors[i].state_dict()
            save_dict[f'actor_target_{i}'] = self.actor_targets[i].state_dict()
        save_dict['critic1'] = self.critic1.state_dict()
        save_dict['critic2'] = self.critic2.state_dict()
        save_dict['critic1_target'] = self.critic1_target.state_dict()
        save_dict['critic2_target'] = self.critic2_target.state_dict()

        torch.save(save_dict, path)
        print(f"模型已保存到 {path}")

    def load(self, path: str):
        """加载模型"""
        if not os.path.exists(path):
            print(f"模型文件 {path} 不存在")
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.num_actors = checkpoint.get('num_actors', self.num_actors)

        for i in range(self.num_actors):
            self.actors[i].load_state_dict(checkpoint[f'actor_{i}'])
            self.actor_targets[i].load_state_dict(checkpoint[f'actor_target_{i}'])
            self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizers'][i])

        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.update_step = checkpoint.get('update_step', 0)
        print(f"已加载 ESAC 模型 {path} ({self.num_actors} actors)")
        return True

    def evaluate(self, env, episodes: int = 5) -> Dict[str, float]:
        """评估当前策略"""
        total_rewards = []
        final_masses = []
        energy_costs = []

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)
            final_masses.append(info.get('current_mass', info.get('final_mass', 0.0)))
            energy_costs.append(info.get('total_energy_cost', 0.0))

        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_final_mass': np.mean(final_masses),
            'mean_energy_cost': np.mean(energy_costs),
        }
