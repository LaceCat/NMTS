"""
True discrete-control baseline for DD mode.

Action set:
0 -> [0, 0]
1 -> [0, 70]
2 -> [50, 0]
3 -> [50, 70]
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import GRUQNetwork

STATE_DIM = 9
NUM_ACTIONS = 4
HIDDEN_DIM = 256
BUFFER_CAPACITY = 200_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50_000


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS, hidden_dim: int = HIDDEN_DIM):
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
            nn.Linear(hidden_dim // 4, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DiscreteReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, state, action_index, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action_index),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int, device: str = "cpu"):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, action_indices, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(action_indices, dtype=np.int64), dtype=torch.long, device=device).unsqueeze(1),
            torch.tensor(np.asarray(rewards, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.asarray(next_states, dtype=np.float32), dtype=torch.float32, device=device),
            torch.tensor(np.asarray(dones, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(1),
        )

    @property
    def size(self) -> int:
        return len(self.buffer)


class DiscreteDDQNAgent:
    """True discrete controller for DD mode."""

    ACTION_TABLE = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 70.0],
            [50.0, 0.0],
            [50.0, 70.0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = 2,
        hidden_dim: int = HIDDEN_DIM,
        buffer_capacity: int = BUFFER_CAPACITY,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        tau: float = TAU,
        lr_actor: float = LR,
        lr_critic: float = LR,
        use_gru_encoder: bool = False,
        gru_hidden_dim: int = 96,
        device: str = "cpu",
    ):
        del action_dim, lr_critic
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_gru_encoder = bool(use_gru_encoder)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.num_actions = int(self.ACTION_TABLE.shape[0])

        q_cls = GRUQNetwork if self.use_gru_encoder else QNetwork
        q_kwargs = {
            "state_dim": state_dim,
            "num_actions": self.num_actions,
            "hidden_dim": hidden_dim,
        }
        if self.use_gru_encoder:
            q_kwargs["gru_hidden_dim"] = self.gru_hidden_dim

        self.q_network = q_cls(**q_kwargs).to(device)
        self.target_q_network = q_cls(**q_kwargs).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        for param in self.target_q_network.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr_actor)
        self.buffer = DiscreteReplayBuffer(capacity=buffer_capacity)

        self.epsilon_start = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay_steps = EPSILON_DECAY_STEPS
        self.select_step = 0
        self.update_step = 0

        self.all_rewards = []
        self.all_final_masses = []
        self.all_energy_costs = []

    def _epsilon(self) -> float:
        ratio = min(float(self.select_step) / max(float(self.epsilon_decay_steps), 1.0), 1.0)
        return float(self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start))

    def _action_index_from_action(self, action: np.ndarray) -> int:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"Expected 2 action values, got shape {np.asarray(action).shape}")
        diffs = np.sum(np.abs(self.ACTION_TABLE - arr[None, :]), axis=1)
        return int(np.argmin(diffs))

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            greedy_index = int(torch.argmax(q_values, dim=1).item())

        if deterministic:
            action_index = greedy_index
        else:
            epsilon = self._epsilon()
            if np.random.rand() < epsilon:
                action_index = int(np.random.randint(self.num_actions))
            else:
                action_index = greedy_index
            self.select_step += 1

        return self.ACTION_TABLE[action_index].copy()

    def store_transition(self, state, action, reward, next_state, done):
        action_index = self._action_index_from_action(action)
        self.buffer.add(state, action_index, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device)
        if batch is None:
            return {}

        states, action_indices, rewards, next_states, dones = batch

        q_values = self.q_network(states).gather(1, action_indices)

        with torch.no_grad():
            next_online_q = self.q_network(next_states)
            next_action_indices = torch.argmax(next_online_q, dim=1, keepdim=True)
            next_target_q = self.target_q_network(next_states).gather(1, next_action_indices)
            q_target = rewards + self.gamma * (1.0 - dones) * next_target_q

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.target_q_network, self.q_network, self.tau)
        self.update_step += 1

        return {
            "critic_loss": float(loss.item()),
            "buffer_size": float(self.buffer.size),
            "epsilon": float(self._epsilon()),
        }

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "select_step": self.select_step,
                "update_step": self.update_step,
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "use_gru_encoder": self.use_gru_encoder,
                "gru_hidden_dim": self.gru_hidden_dim,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.select_step = int(checkpoint.get("select_step", 0))
        self.update_step = int(checkpoint.get("update_step", 0))
        print(f"Model loaded from {path}")
        return True

    def evaluate(self, env, episodes: int = 5) -> Dict[str, float]:
        total_rewards = []
        final_masses = []
        energy_costs = []

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0.0
            info: Optional[dict] = None

            while not done:
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            info = info or {}
            total_rewards.append(episode_reward)
            final_masses.append(info.get("current_mass", info.get("final_mass", 0.0)))
            energy_costs.append(info.get("total_energy_cost", 0.0))

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_final_mass": float(np.mean(final_masses)),
            "mean_energy_cost": float(np.mean(energy_costs)),
        }
