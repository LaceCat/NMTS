"""
Standard single-actor TD3 baseline.

Reference:
Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", 2018.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import Actor, Critic
from utils.replay_buffer import ReplayBuffer


STATE_DIM = 9
ACTION_DIM = 2
HIDDEN_DIM = 256
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
POLICY_NOISE = 0.10
NOISE_CLIP = 0.20
POLICY_FREQ = 2
EXPLORATION_NOISE = 0.10
ACTION_BOUNDS_LOW = np.array([0.0, 0.0], dtype=np.float32)
ACTION_BOUNDS_HIGH = np.array([50.0, 70.0], dtype=np.float32)


class TD3Agent:
    """Single-actor TD3 agent with twin critics and uniform replay."""

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
        policy_noise: float = POLICY_NOISE,
        noise_clip: float = NOISE_CLIP,
        policy_freq: int = POLICY_FREQ,
        exploration_noise: float = EXPLORATION_NOISE,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = int(policy_freq)
        self.device = device
        self.update_step = 0

        self.action_low_np = ACTION_BOUNDS_LOW.copy()
        self.action_high_np = ACTION_BOUNDS_HIGH.copy()
        self.action_range_np = self.action_high_np - self.action_low_np

        self.action_low = torch.tensor(self.action_low_np, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(self.action_high_np, dtype=torch.float32, device=device)

        self.policy_noise_np = self._resolve_noise(policy_noise, default_fraction=0.10)
        self.noise_clip_np = self._resolve_noise(noise_clip, default_fraction=0.20)
        self.exploration_noise_np = self._resolve_noise(exploration_noise, default_fraction=0.10)

        self.policy_noise = torch.tensor(self.policy_noise_np, dtype=torch.float32, device=device)
        self.noise_clip = torch.tensor(self.noise_clip_np, dtype=torch.float32, device=device)

        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.all_rewards = []
        self.all_final_masses = []
        self.all_energy_costs = []

    def _resolve_noise(self, value, default_fraction: float) -> np.ndarray:
        if value is None:
            return self.action_range_np * default_fraction

        noise = np.asarray(value, dtype=np.float32)
        if noise.ndim == 0:
            scalar = float(noise)
            if scalar <= 1.0:
                return self.action_range_np * scalar
            return np.full(self.action_dim, scalar, dtype=np.float32)

        if noise.shape != (self.action_dim,):
            raise ValueError(f"Noise shape must be ({self.action_dim},), got {noise.shape}")

        if np.all(noise <= 1.0):
            return self.action_range_np * noise
        return noise.astype(np.float32)

    def _scale_action(self, raw_action: np.ndarray) -> np.ndarray:
        return 0.5 * self.action_range_np * (raw_action + 1.0) + self.action_low_np

    def _scale_action_tensor(self, raw_action: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.action_high - self.action_low) * (raw_action + 1.0) + self.action_low

    def _clip_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def _clip_noise_tensor(self, noise: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(noise, self.noise_clip), -self.noise_clip)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.actor(state_tensor).cpu().numpy()[0]

        action = self._scale_action(raw_action)

        if not deterministic:
            action = action + np.random.normal(
                loc=0.0,
                scale=self.exploration_noise_np,
                size=self.action_dim,
            ).astype(np.float32)

        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device)
        if batch is None:
            return {}

        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = self._clip_noise_tensor(noise)

            next_actions = self._scale_action_tensor(self.actor_target(next_states))
            next_actions = self._clip_action_tensor(next_actions + noise)

            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            q_target = rewards + self.gamma * (1.0 - dones) * target_q

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss_value = 0.0
        if self.update_step % self.policy_freq == 0:
            new_actions = self._scale_action_tensor(self.actor(states))
            actor_loss = -self.critic1(states, new_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor, self.tau)
            self._soft_update(self.critic1_target, self.critic1, self.tau)
            self._soft_update(self.critic2_target, self.critic2, self.tau)
            actor_loss_value = float(actor_loss.item())

        self.update_step += 1

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": actor_loss_value,
            "buffer_size": float(self.buffer.size),
        }

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "update_step": self.update_step,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])

        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.critic1_target.load_state_dict(checkpoint.get("critic1_target", checkpoint["critic1"]))
        self.critic2_target.load_state_dict(checkpoint.get("critic2_target", checkpoint["critic2"]))

        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic1_optimizer" in checkpoint:
            self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        if "critic2_optimizer" in checkpoint:
            self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

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

            total_rewards.append(episode_reward)
            info = info or {}
            final_masses.append(info.get("current_mass", info.get("final_mass", 0.0)))
            energy_costs.append(info.get("total_energy_cost", 0.0))

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_final_mass": float(np.mean(final_masses)),
            "mean_energy_cost": float(np.mean(energy_costs)),
        }
