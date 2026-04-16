"""
神经网络定义 - TD3 Actor-Critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor 网络 - 输出连续动作

    结构: Linear(256) -> LayerNorm -> ReLU -> Linear(128) -> LayerNorm -> ReLU -> Linear(64) -> LayerNorm -> ReLU -> Linear(2) -> Tanh
    """

    def __init__(self, state_dim: int = 9, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        return torch.tanh(self.head(x))


class Critic(nn.Module):
    """
    Critic 网络 - 输出状态-动作对的价值

    结构: Linear(state+action, 256) -> LayerNorm -> ReLU -> Linear(128) -> LayerNorm -> ReLU -> Linear(64) -> LayerNorm -> ReLU -> Linear(1)
    """

    def __init__(self, state_dim: int = 9, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
