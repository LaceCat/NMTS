"""
Neural network modules for TD3 actor-critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Standard feed-forward actor."""

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
    """Standard feed-forward critic."""

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


class HistoryGRUEncoder(nn.Module):
    """
    Encode the augmented observation with a GRU over short state history.

    Expected observation layout:
    - current base state: 9 dims
    - action history: 6 dims
    - past base-state history: 3 x 9 dims
    - optional extra dims appended after that
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 96,
        base_obs_dim: int = 9,
        action_history_steps: int = 3,
        state_history_steps: int = 3,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.base_obs_dim = int(base_obs_dim)
        self.action_history_steps = int(action_history_steps)
        self.state_history_steps = int(state_history_steps)
        self.action_history_dim = self.action_history_steps * 2
        self.state_history_dim = self.state_history_steps * self.base_obs_dim
        self.core_state_dim = self.base_obs_dim + self.action_history_dim + self.state_history_dim
        if self.state_dim < self.core_state_dim:
            raise ValueError(
                f"HistoryGRUEncoder expects at least {self.core_state_dim} dims, got {self.state_dim}"
            )

        self.extra_dim = self.state_dim - self.core_state_dim
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru = nn.GRU(
            input_size=self.base_obs_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        fusion_dim = self.gru_hidden_dim + self.base_obs_dim + self.action_history_dim + self.extra_dim
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim // 4

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        current = state[:, : self.base_obs_dim]
        action_hist_start = self.base_obs_dim
        action_hist_end = action_hist_start + self.action_history_dim
        action_history = state[:, action_hist_start:action_hist_end]

        hist_start = action_hist_end
        hist_end = hist_start + self.state_history_dim
        history_flat = state[:, hist_start:hist_end]
        past_states = history_flat.view(-1, self.state_history_steps, self.base_obs_dim)
        sequence = torch.cat([past_states, current.unsqueeze(1)], dim=1)
        _, hidden = self.gru(sequence)
        gru_feature = hidden[-1]

        if self.extra_dim > 0:
            extra = state[:, hist_end:]
            fused = torch.cat([gru_feature, current, action_history, extra], dim=-1)
        else:
            fused = torch.cat([gru_feature, current, action_history], dim=-1)

        return self.net(fused)


class GRUActor(nn.Module):
    """TD3 actor with a short-history GRU encoder."""

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 2,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 96,
    ):
        super().__init__()
        self.encoder = HistoryGRUEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )
        self.head = nn.Linear(self.encoder.output_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.encoder(state)
        return torch.tanh(self.head(x))


class GRUCritic(nn.Module):
    """TD3 critic with a short-history GRU encoder."""

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 2,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 96,
    ):
        super().__init__()
        self.encoder = HistoryGRUEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )
        critic_hidden = max(hidden_dim // 2, 64)
        critic_mid = max(hidden_dim // 4, 32)
        self.q_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + action_dim, critic_hidden),
            nn.LayerNorm(critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, critic_mid),
            nn.LayerNorm(critic_mid),
            nn.ReLU(),
            nn.Linear(critic_mid, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_feature = self.encoder(state)
        x = torch.cat([state_feature, action], dim=-1)
        return self.q_head(x)


class GRUHybridActor(nn.Module):
    """Hybrid actor with a short-history GRU encoder."""

    def __init__(
        self,
        state_dim: int = 42,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 96,
    ):
        super().__init__()
        self.encoder = HistoryGRUEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )
        self.q_uf_head = nn.Linear(self.encoder.output_dim, 1)
        self.q_fp_logit_head = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, state: torch.Tensor):
        x = self.encoder(state)
        q_uf_raw = torch.tanh(self.q_uf_head(x))
        q_fp_logit = self.q_fp_logit_head(x)
        return q_uf_raw, q_fp_logit


class GRUHybridCritic(nn.Module):
    """Hybrid critic with a short-history GRU encoder."""

    def __init__(
        self,
        state_dim: int = 42,
        action_dim: int = 2,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 96,
    ):
        super().__init__()
        self.encoder = HistoryGRUEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )
        critic_hidden = max(hidden_dim // 2, 64)
        critic_mid = max(hidden_dim // 4, 32)
        self.q_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + action_dim, critic_hidden),
            nn.LayerNorm(critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, critic_mid),
            nn.LayerNorm(critic_mid),
            nn.ReLU(),
            nn.Linear(critic_mid, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_feature = self.encoder(state)
        x = torch.cat([state_feature, action], dim=-1)
        return self.q_head(x)


class GRUQNetwork(nn.Module):
    """Discrete Q-network with a short-history GRU encoder."""

    def __init__(
        self,
        state_dim: int = 42,
        num_actions: int = 4,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 96,
    ):
        super().__init__()
        self.encoder = HistoryGRUEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
        )
        self.head = nn.Linear(self.encoder.output_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.encoder(state)
        return self.head(x)
