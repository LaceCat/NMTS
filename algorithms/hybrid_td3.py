"""
True hybrid controller for CD mode.

- Q_uf: continuous in [0, 50]
- Q_fp: discrete in {0, 70}

The actor outputs a continuous underflow command and a Bernoulli logit for the
filter-press switch. The critics evaluate actual physical actions.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import GRUHybridActor, GRUHybridCritic
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
DISCRETE_EXPLORATION_PROB = 0.10


class HybridActor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = HIDDEN_DIM):
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
        self.q_uf_head = nn.Linear(hidden_dim // 4, 1)
        self.q_fp_logit_head = nn.Linear(hidden_dim // 4, 1)

    def forward(self, state: torch.Tensor):
        x = self.net(state)
        q_uf_raw = torch.tanh(self.q_uf_head(x))
        q_fp_logit = self.q_fp_logit_head(x)
        return q_uf_raw, q_fp_logit


class HybridCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = HIDDEN_DIM):
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
        return self.net(torch.cat([state, action], dim=-1))


class HybridTD3Agent:
    """True mixed continuous-discrete controller for CD mode."""

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
        discrete_exploration_prob: float = DISCRETE_EXPLORATION_PROB,
        q_uf_low: float = 0.0,
        q_uf_high: float = 50.0,
        use_gru_encoder: bool = False,
        gru_hidden_dim: int = 96,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_gru_encoder = bool(use_gru_encoder)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = int(policy_freq)
        self.device = device
        self.update_step = 0

        self.q_uf_low = float(q_uf_low)
        self.q_uf_high = float(q_uf_high)
        self.q_fp_off = 0.0
        self.q_fp_on = 70.0

        q_uf_range = self.q_uf_high - self.q_uf_low
        self.policy_noise_np = np.asarray([q_uf_range * float(policy_noise if policy_noise <= 1.0 else policy_noise)], dtype=np.float32)
        self.noise_clip_np = np.asarray([q_uf_range * float(noise_clip if noise_clip <= 1.0 else noise_clip)], dtype=np.float32)
        self.exploration_noise_np = np.asarray([q_uf_range * float(exploration_noise if exploration_noise <= 1.0 else exploration_noise)], dtype=np.float32)
        self.discrete_exploration_prob = float(discrete_exploration_prob)

        self.policy_noise = torch.tensor(self.policy_noise_np, dtype=torch.float32, device=device)
        self.noise_clip = torch.tensor(self.noise_clip_np, dtype=torch.float32, device=device)

        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        actor_cls = GRUHybridActor if self.use_gru_encoder else HybridActor
        critic_cls = GRUHybridCritic if self.use_gru_encoder else HybridCritic
        actor_kwargs = {
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
        }
        critic_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
        }
        if self.use_gru_encoder:
            actor_kwargs["gru_hidden_dim"] = self.gru_hidden_dim
            critic_kwargs["gru_hidden_dim"] = self.gru_hidden_dim

        self.actor = actor_cls(**actor_kwargs).to(device)
        self.actor_target = actor_cls(**actor_kwargs).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = critic_cls(**critic_kwargs).to(device)
        self.critic2 = critic_cls(**critic_kwargs).to(device)
        self.critic1_target = critic_cls(**critic_kwargs).to(device)
        self.critic2_target = critic_cls(**critic_kwargs).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for net in [self.actor_target, self.critic1_target, self.critic2_target]:
            for param in net.parameters():
                param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.all_rewards = []
        self.all_final_masses = []
        self.all_energy_costs = []

    def _scale_q_uf_tensor(self, raw_q_uf: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.q_uf_high - self.q_uf_low) * (raw_q_uf + 1.0) + self.q_uf_low

    def _build_action_tensor(self, q_uf: torch.Tensor, q_fp_binary: torch.Tensor) -> torch.Tensor:
        q_fp = self.q_fp_on * q_fp_binary
        return torch.cat([q_uf, q_fp], dim=-1)

    def _target_expected_q(self, states: torch.Tensor) -> torch.Tensor:
        next_q_uf_raw, next_q_fp_logit = self.actor_target(states)
        next_q_uf = self._scale_q_uf_tensor(next_q_uf_raw)

        noise = torch.randn_like(next_q_uf) * self.policy_noise
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_q_uf = torch.clamp(next_q_uf + noise, self.q_uf_low, self.q_uf_high)

        next_q_fp_prob = torch.sigmoid(next_q_fp_logit)
        action_off = self._build_action_tensor(next_q_uf, torch.zeros_like(next_q_fp_prob))
        action_on = self._build_action_tensor(next_q_uf, torch.ones_like(next_q_fp_prob))

        q_off = torch.min(
            self.critic1_target(states, action_off),
            self.critic2_target(states, action_off),
        )
        q_on = torch.min(
            self.critic1_target(states, action_on),
            self.critic2_target(states, action_on),
        )
        return (1.0 - next_q_fp_prob) * q_off + next_q_fp_prob * q_on

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_uf_raw, q_fp_logit = self.actor(state_tensor)
            q_uf = float(self._scale_q_uf_tensor(q_uf_raw).cpu().numpy()[0, 0])
            q_fp_prob = float(torch.sigmoid(q_fp_logit).cpu().numpy()[0, 0])

        if deterministic:
            q_fp_binary = 1.0 if q_fp_prob >= 0.5 else 0.0
        else:
            q_uf = float(q_uf + np.random.normal(0.0, self.exploration_noise_np[0]))
            q_uf = float(np.clip(q_uf, self.q_uf_low, self.q_uf_high))

            if np.random.rand() < self.discrete_exploration_prob:
                q_fp_binary = float(np.random.randint(0, 2))
            else:
                q_fp_binary = float(np.random.rand() < q_fp_prob)

        q_fp = self.q_fp_on if q_fp_binary > 0.5 else self.q_fp_off
        return np.asarray([q_uf, q_fp], dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device)
        if batch is None:
            return {}

        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            expected_next_q = self._target_expected_q(next_states)
            q_target = rewards + self.gamma * (1.0 - dones) * expected_next_q

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
            q_uf_raw, q_fp_logit = self.actor(states)
            q_uf = self._scale_q_uf_tensor(q_uf_raw)
            q_fp_prob = torch.sigmoid(q_fp_logit)

            action_off = self._build_action_tensor(q_uf, torch.zeros_like(q_fp_prob))
            action_on = self._build_action_tensor(q_uf, torch.ones_like(q_fp_prob))
            q_off = self.critic1(states, action_off)
            q_on = self.critic1(states, action_on)
            expected_q = (1.0 - q_fp_prob) * q_off + q_fp_prob * q_on
            actor_loss = -expected_q.mean()

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
                "state_dim": self.state_dim,
                "hidden_dim": self.hidden_dim,
                "use_gru_encoder": self.use_gru_encoder,
                "gru_hidden_dim": self.gru_hidden_dim,
                "q_uf_low": self.q_uf_low,
                "q_uf_high": self.q_uf_high,
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
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
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
