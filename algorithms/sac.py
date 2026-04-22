"""
Standard single-actor Soft Actor-Critic for the continuous-control CC setup.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import HistoryGRUEncoder
from utils.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


STATE_DIM = 42
ACTION_DIM = 2
HIDDEN_DIM = 256
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
INIT_ALPHA = 0.2
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
MEAN_ACTION_Q_WEIGHT = 0.35
STD_REG_WEIGHT = 0.02
ACTION_BOUNDS_LOW = np.array([0.0, 0.0], dtype=np.float32)
ACTION_BOUNDS_HIGH = np.array([50.0, 70.0], dtype=np.float32)
PER_ALPHA = 0.0
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_FRAMES = 100000
N_STEP = 1
REWARD_SCALE = 0.01


def _safe_atanh(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -0.999, 0.999)
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
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
        self.mean_head = nn.Linear(hidden_dim // 4, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class GaussianCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
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


class GRUGaussianActor(nn.Module):
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
        self.mean_head = nn.Linear(self.encoder.output_dim, action_dim)
        self.log_std_head = nn.Linear(self.encoder.output_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(state)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std


class GRUGaussianCritic(nn.Module):
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
        feat = self.encoder(state)
        return self.q_head(torch.cat([feat, action], dim=-1))


class SACAgent:
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
        lr_alpha: float = LR_ALPHA,
        init_alpha: float = INIT_ALPHA,
        target_entropy: Optional[float] = None,
        action_low=None,
        action_high=None,
        use_gru_encoder: bool = False,
        gru_hidden_dim: int = 96,
        mean_action_q_weight: float = MEAN_ACTION_Q_WEIGHT,
        std_reg_weight: float = STD_REG_WEIGHT,
        n_step: int = N_STEP,
        per_alpha: float = PER_ALPHA,
        reward_scale: float = REWARD_SCALE,
        use_actor_prior: bool = False,
        behavior_clone_weight: float = 0.0,
        behavior_clone_q_uf_weight: float = 0.35,
        behavior_clone_q_fp_weight: float = 1.0,
        teacher_action_weight: float = 0.0,
        teacher_action_q_uf_weight: float = 1.0,
        teacher_action_q_fp_weight: float = 1.0,
        q_fp_teacher_weight: float = 0.0,
        q_fp_teacher_residual_threshold: float = 2.0,
        q_fp_teacher_decision_interval: int = 5,
        q_fp_teacher_q_fp_delta_max: float = -1.0,
        q_uf_is_delta: bool = False,
        device: str = "cpu",
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = device
        self.use_gru_encoder = bool(use_gru_encoder)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.mean_action_q_weight = float(mean_action_q_weight)
        self.std_reg_weight = float(std_reg_weight)
        self.n_step = max(int(n_step), 1)
        self.per_alpha = float(per_alpha)
        self.reward_scale = float(reward_scale)
        self.use_actor_prior = bool(use_actor_prior)
        self.behavior_clone_weight = float(max(behavior_clone_weight, 0.0))
        self.behavior_clone_q_uf_weight = float(max(behavior_clone_q_uf_weight, 0.0))
        self.behavior_clone_q_fp_weight = float(max(behavior_clone_q_fp_weight, 0.0))
        self.teacher_action_weight = float(max(teacher_action_weight, 0.0))
        self.teacher_action_q_uf_weight = float(max(teacher_action_q_uf_weight, 0.0))
        self.teacher_action_q_fp_weight = float(max(teacher_action_q_fp_weight, 0.0))
        self.q_fp_teacher_weight = float(max(q_fp_teacher_weight, 0.0))
        self.q_fp_teacher_residual_threshold = float(max(q_fp_teacher_residual_threshold, 0.0))
        self.q_fp_teacher_decision_interval = int(max(q_fp_teacher_decision_interval, 1))
        self.q_fp_teacher_q_fp_delta_max = (
            None if q_fp_teacher_q_fp_delta_max is None or q_fp_teacher_q_fp_delta_max < 0
            else float(q_fp_teacher_q_fp_delta_max)
        )
        self.q_uf_is_delta = bool(q_uf_is_delta)
        self.actor_only_update = False
        self.freeze_alpha_update = False
        self.distill_only_update = False
        self.update_step = 0

        self.action_low_np = (
            np.asarray(action_low, dtype=np.float32).copy()
            if action_low is not None else ACTION_BOUNDS_LOW.copy()
        )
        self.action_high_np = (
            np.asarray(action_high, dtype=np.float32).copy()
            if action_high is not None else ACTION_BOUNDS_HIGH.copy()
        )
        self.action_range_np = self.action_high_np - self.action_low_np
        self.action_bias_np = 0.5 * (self.action_high_np + self.action_low_np)

        self.action_low = torch.tensor(self.action_low_np, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(self.action_high_np, dtype=torch.float32, device=device)
        self.action_scale = torch.tensor(0.5 * self.action_range_np, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor(self.action_bias_np, dtype=torch.float32, device=device)
        self.log_action_scale_sum = torch.log(self.action_scale).sum()

        if self.per_alpha > 0.0:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                alpha=self.per_alpha,
                beta_start=PER_BETA_START,
                beta_end=PER_BETA_END,
                beta_frames=PER_BETA_FRAMES,
                n_step=self.n_step,
                gamma=self.gamma,
            )
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity)

        actor_cls = GRUGaussianActor if self.use_gru_encoder else GaussianActor
        critic_cls = GRUGaussianCritic if self.use_gru_encoder else GaussianCritic

        actor_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }
        critic_kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }
        if self.use_gru_encoder:
            actor_kwargs["gru_hidden_dim"] = self.gru_hidden_dim
            critic_kwargs["gru_hidden_dim"] = self.gru_hidden_dim

        self.actor = actor_cls(**actor_kwargs).to(device)
        self.critic1 = critic_cls(**critic_kwargs).to(device)
        self.critic2 = critic_cls(**critic_kwargs).to(device)
        self.critic1_target = critic_cls(**critic_kwargs).to(device)
        self.critic2_target = critic_cls(**critic_kwargs).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        if self.use_actor_prior:
            self._init_actor_prior()

        for net in (self.critic1_target, self.critic2_target):
            for param in net.parameters():
                param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)

        if target_entropy is None:
            target_entropy = -float(self.action_dim)
        self.target_entropy = float(target_entropy)
        self.auto_alpha = bool(lr_alpha > 0.0)
        self.log_alpha = torch.tensor(
            np.log(max(init_alpha, 1e-6)),
            dtype=torch.float32,
            device=device,
            requires_grad=self.auto_alpha,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=lr_alpha)
            if self.auto_alpha else None
        )

        self.all_rewards = []
        self.all_final_masses = []
        self.all_energy_costs = []

    def _init_actor_prior(self):
        # Standard SAC in this project starts interacting with the environment
        # immediately; it does not use a separate random-action warmup policy.
        # If the policy mean starts at zero, the squashed action maps to the
        # middle of each physical range (e.g. Q_uf≈25), which matches the bad
        # steady state we repeatedly observed. Bias the initial policy toward a
        # low-flow operating prior instead.
        if not hasattr(self.actor, "mean_head") or not hasattr(self.actor, "log_std_head"):
            return

        # If the first action dimension is signed, SAC is running with delta
        # control on Q_uf. For scratch curriculum runs we want the initial
        # deterministic policy to pull concentration back toward the safe band,
        # so bias the prior toward a positive underflow ramp while keeping
        # Q_fp almost closed. Otherwise bias the absolute underflow toward a
        # modest but non-trivial draw.
        if self.action_low_np[0] < 0.0:
            desired_action = np.array(
                [
                    self.action_low_np[0] + self.action_range_np[0] * 0.58,
                    self.action_low_np[1] + self.action_range_np[1] * 0.26,
                ],
                dtype=np.float32,
            )
        else:
            desired_action = self.action_low_np + self.action_range_np * np.array([0.30, 0.07], dtype=np.float32)
        desired_norm = 2.0 * (desired_action - self.action_low_np) / np.maximum(self.action_range_np, 1e-6) - 1.0
        desired_pre_tanh = _safe_atanh(desired_norm)

        with torch.no_grad():
            nn.init.zeros_(self.actor.mean_head.weight)
            self.actor.mean_head.bias.copy_(
                torch.tensor(desired_pre_tanh, dtype=torch.float32, device=self.device)
            )
            nn.init.zeros_(self.actor.log_std_head.weight)
            self.actor.log_std_head.bias.fill_(-2.0)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean, log_std = self.actor(state)
        if deterministic:
            z = mean
            squashed = torch.tanh(z)
            action = squashed * self.action_scale + self.action_bias
            return action, None

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.action_scale + self.action_bias

        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= self.log_action_scale_sum
        return action, log_prob

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self._sample_action(state_tensor, deterministic=deterministic)
        return action.cpu().numpy()[0].astype(np.float32)

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        teacher_action=None,
        teacher_active: bool = False,
    ):
        scaled_reward = float(reward) * self.reward_scale
        self.buffer.add(
            state,
            action,
            scaled_reward,
            next_state,
            done,
            teacher_action=teacher_action,
            teacher_active=teacher_active,
        )

    def _compute_q_fp_teacher_loss(
        self,
        states: torch.Tensor,
        det_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = det_actions.new_tensor(0.0)
        if self.q_fp_teacher_weight <= 0.0:
            return zero, zero
        if states.dim() != 2 or states.shape[1] < 15 or det_actions.shape[1] < 2:
            return zero, zero

        # For transformed SAC observations:
        # - states[:, 1]   = normalized V_buf
        # - states[:, 13]  = latest previous Q_uf / 50
        # - states[:, 14]  = latest previous Q_fp / 70
        v_buf = torch.clamp(states[:, 1], 0.0, 1.0) * 30.0
        prev_q_uf = torch.clamp(states[:, 13], 0.0, 1.0) * 50.0
        prev_q_fp = torch.clamp(states[:, 14], 0.0, 1.0) * 70.0

        if self.q_uf_is_delta:
            q_uf = torch.clamp(prev_q_uf + det_actions[:, 0], 0.0, 50.0)
        else:
            q_uf = torch.clamp(det_actions[:, 0], 0.0, 50.0)
        q_fp = torch.clamp(det_actions[:, 1], 0.0, 70.0)

        projected_residual = v_buf + self.q_fp_teacher_decision_interval * (q_uf - q_fp) / 60.0
        teacher_mask = (projected_residual > 0.0) & (
            projected_residual <= self.q_fp_teacher_residual_threshold
        )
        if not bool(torch.any(teacher_mask).item()):
            return zero, zero

        teacher_q_fp = q_uf + 60.0 * v_buf / float(self.q_fp_teacher_decision_interval)
        teacher_q_fp = torch.clamp(teacher_q_fp, 0.0, 70.0)
        if self.q_fp_teacher_q_fp_delta_max is not None:
            lower = torch.clamp(prev_q_fp - self.q_fp_teacher_q_fp_delta_max, 0.0, 70.0)
            upper = torch.clamp(prev_q_fp + self.q_fp_teacher_q_fp_delta_max, 0.0, 70.0)
            teacher_q_fp = torch.max(torch.min(teacher_q_fp, upper), lower)

        teacher_weight = teacher_mask.float()
        teacher_loss = ((q_fp - teacher_q_fp).pow(2) * teacher_weight).sum() / teacher_weight.sum().clamp_min(1.0)
        teacher_active_rate = teacher_weight.mean()
        return teacher_loss, teacher_active_rate

    def _compute_teacher_action_loss(
        self,
        det_actions: torch.Tensor,
        teacher_actions: Optional[torch.Tensor],
        teacher_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = det_actions.new_tensor(0.0)
        if self.teacher_action_weight <= 0.0:
            return zero, zero
        if teacher_actions is None or teacher_mask is None:
            return zero, zero
        if teacher_actions.shape != det_actions.shape:
            return zero, zero

        mask = teacher_mask.view(-1, 1).to(det_actions.dtype)
        if not bool(torch.any(mask > 0.0).item()):
            return zero, zero

        teacher_weights = torch.tensor(
            [self.teacher_action_q_uf_weight, self.teacher_action_q_fp_weight],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        # Normalize by physical action ranges so the Q_fp term does not dwarf
        # the SAC objective simply because it lives on a much larger numeric
        # scale than delta-Q_uf.
        action_range = (self.action_high - self.action_low).clamp_min(1e-6).view(1, -1)
        normalized_sq_error = ((det_actions - teacher_actions) / action_range).pow(2)
        teacher_loss = ((normalized_sq_error * teacher_weights) * mask).sum() / mask.sum().clamp_min(1.0)
        teacher_active_rate = mask.mean()
        return teacher_loss, teacher_active_rate

    def update(self) -> Dict[str, float]:
        batch = self.buffer.sample(self.batch_size, self.device, return_teacher=True)
        if batch is None:
            return {}

        teacher_actions = None
        teacher_mask = None
        if len(batch) == 9:
            states, actions, rewards, next_states, dones, indices, weights, teacher_actions, teacher_mask = batch
        elif len(batch) == 7:
            states, actions, rewards, next_states, dones, teacher_actions, teacher_mask = batch
            indices = None
            weights = torch.ones_like(rewards)
        else:
            states, actions, rewards, next_states, dones = batch
            indices = None
            weights = torch.ones_like(rewards)

        td_error1 = None
        td_error2 = None
        critic1_loss = states.new_tensor(0.0)
        critic2_loss = states.new_tensor(0.0)

        if not self.actor_only_update:
            with torch.no_grad():
                next_actions, next_log_prob = self._sample_action(next_states, deterministic=False)
                target_q1 = self.critic1_target(next_states, next_actions)
                target_q2 = self.critic2_target(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
                gamma_n = self.gamma ** self.n_step
                q_target = rewards + gamma_n * (1.0 - dones) * target_q

            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            td_error1 = q1 - q_target
            td_error2 = q2 - q_target
            critic1_loss = (weights * td_error1.pow(2)).mean()
            critic2_loss = (weights * td_error2.pow(2)).mean()

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

        mean, log_std = self.actor(states)
        new_actions, log_prob = self._sample_action(states, deterministic=False)
        distill_only = bool(self.actor_only_update and self.distill_only_update)
        if self.actor_only_update:
            for param in self.critic1.parameters():
                param.requires_grad_(False)
            for param in self.critic2.parameters():
                param.requires_grad_(False)
        try:
            det_actions = torch.tanh(mean) * self.action_scale + self.action_bias
            std_reg_loss = log_std.exp().pow(2).mean()
            bc_weights = torch.tensor(
                [self.behavior_clone_q_uf_weight, self.behavior_clone_q_fp_weight],
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)
            behavior_clone_loss = ((det_actions - actions).pow(2) * bc_weights).mean()
            teacher_action_loss, teacher_action_active_rate = self._compute_teacher_action_loss(
                det_actions,
                teacher_actions,
                teacher_mask,
            )
            q_fp_teacher_loss, q_fp_teacher_active_rate = self._compute_q_fp_teacher_loss(states, det_actions)
            if distill_only:
                mean_action_loss = det_actions.new_tensor(0.0)
                distill_terms = []
                if self.behavior_clone_weight > 0.0:
                    distill_terms.append(self.behavior_clone_weight * behavior_clone_loss)
                if self.teacher_action_weight > 0.0 and bool(teacher_action_active_rate.item() > 0.0):
                    distill_terms.append(self.teacher_action_weight * teacher_action_loss)
                if self.q_fp_teacher_weight > 0.0 and bool(q_fp_teacher_active_rate.item() > 0.0):
                    distill_terms.append(self.q_fp_teacher_weight * q_fp_teacher_loss)
                if distill_terms:
                    actor_loss = sum(distill_terms)
                else:
                    actor_loss = det_actions.new_tensor(0.0)
            else:
                q_new = torch.min(self.critic1(states, new_actions), self.critic2(states, new_actions))
                actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
                q_det = torch.min(self.critic1(states, det_actions), self.critic2(states, det_actions))
                mean_action_loss = -q_det.mean()
                actor_loss = (
                    actor_loss
                    + self.mean_action_q_weight * mean_action_loss
                    + self.std_reg_weight * std_reg_loss
                    + self.behavior_clone_weight * behavior_clone_loss
                    + self.teacher_action_weight * teacher_action_loss
                    + self.q_fp_teacher_weight * q_fp_teacher_loss
                )
        finally:
            if self.actor_only_update:
                for param in self.critic1.parameters():
                    param.requires_grad_(True)
                for param in self.critic2.parameters():
                    param.requires_grad_(True)

        skipped_distill_update = bool(
            distill_only
            and abs(float(actor_loss.item())) <= 1e-12
            and self.behavior_clone_weight <= 0.0
        )
        if not skipped_distill_update:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if distill_only:
                if hasattr(self.actor, "backbone"):
                    for param in self.actor.backbone.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                if hasattr(self.actor, "encoder"):
                    for param in self.actor.encoder.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                if hasattr(self.actor, "mean_head"):
                    if getattr(self.actor.mean_head, "weight", None) is not None and self.actor.mean_head.weight.grad is not None:
                        if self.actor.mean_head.weight.grad.shape[0] >= 1:
                            self.actor.mean_head.weight.grad[0].zero_()
                    if getattr(self.actor.mean_head, "bias", None) is not None and self.actor.mean_head.bias.grad is not None:
                        if self.actor.mean_head.bias.grad.shape[0] >= 1:
                            self.actor.mean_head.bias.grad[0].zero_()
                if hasattr(self.actor, "log_std_head"):
                    if getattr(self.actor.log_std_head, "weight", None) is not None and self.actor.log_std_head.weight.grad is not None:
                        self.actor.log_std_head.weight.grad.zero_()
                    if getattr(self.actor.log_std_head, "bias", None) is not None and self.actor.log_std_head.bias.grad is not None:
                        self.actor.log_std_head.bias.grad.zero_()
            self.actor_optimizer.step()

        alpha_loss_value = 0.0
        if self.auto_alpha and self.alpha_optimizer is not None and not self.freeze_alpha_update and not self.actor_only_update:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        if indices is not None and hasattr(self.buffer, "update_priorities") and td_error1 is not None and td_error2 is not None:
            td_errors = 0.5 * (td_error1.detach().abs() + td_error2.detach().abs())
            self.buffer.update_priorities(indices, td_errors.squeeze(1).cpu().numpy())

        if not self.actor_only_update:
            self._soft_update(self.critic1_target, self.critic1, self.tau)
            self._soft_update(self.critic2_target, self.critic2, self.tau)

        self.update_step += 1

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mean_action_loss": float(mean_action_loss.item()),
            "std_reg_loss": float(std_reg_loss.item()),
            "behavior_clone_loss": float(behavior_clone_loss.item()),
            "behavior_clone_weight": float(self.behavior_clone_weight),
            "teacher_action_loss": float(teacher_action_loss.item()),
            "teacher_action_weight": float(self.teacher_action_weight),
            "teacher_action_active_rate": float(teacher_action_active_rate.item()),
            "q_fp_teacher_loss": float(q_fp_teacher_loss.item()),
            "q_fp_teacher_weight": float(self.q_fp_teacher_weight),
            "q_fp_teacher_active_rate": float(q_fp_teacher_active_rate.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().item()),
            "actor_only_update": float(self.actor_only_update),
            "distill_only_update": float(self.distill_only_update),
            "skipped_distill_update": float(skipped_distill_update),
            "buffer_size": float(self.buffer.size),
            "per_beta": float(getattr(self.buffer, "_beta", lambda _: 1.0)(getattr(self.buffer, "frame", 1))),
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
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer is not None else None,
                "log_alpha": float(self.log_alpha.detach().item()),
                "target_entropy": float(self.target_entropy),
                "auto_alpha": bool(self.auto_alpha),
                "update_step": int(self.update_step),
                "state_dim": int(self.state_dim),
                "hidden_dim": int(self.hidden_dim),
                "use_gru_encoder": bool(self.use_gru_encoder),
                "gru_hidden_dim": int(self.gru_hidden_dim),
                "mean_action_q_weight": float(self.mean_action_q_weight),
                "std_reg_weight": float(self.std_reg_weight),
                "n_step": int(self.n_step),
                "per_alpha": float(self.per_alpha),
                "reward_scale": float(self.reward_scale),
                "use_actor_prior": bool(self.use_actor_prior),
                "behavior_clone_weight": float(self.behavior_clone_weight),
                "behavior_clone_q_uf_weight": float(self.behavior_clone_q_uf_weight),
                "behavior_clone_q_fp_weight": float(self.behavior_clone_q_fp_weight),
                "teacher_action_weight": float(self.teacher_action_weight),
                "teacher_action_q_uf_weight": float(self.teacher_action_q_uf_weight),
                "teacher_action_q_fp_weight": float(self.teacher_action_q_fp_weight),
                "q_fp_teacher_weight": float(self.q_fp_teacher_weight),
                "q_fp_teacher_residual_threshold": float(self.q_fp_teacher_residual_threshold),
                "q_fp_teacher_decision_interval": int(self.q_fp_teacher_decision_interval),
                "q_fp_teacher_q_fp_delta_max": (
                    None if self.q_fp_teacher_q_fp_delta_max is None else float(self.q_fp_teacher_q_fp_delta_max)
                ),
                "q_uf_is_delta": bool(self.q_uf_is_delta),
                "actor_only_update": bool(self.actor_only_update),
                "freeze_alpha_update": bool(self.freeze_alpha_update),
                "distill_only_update": bool(self.distill_only_update),
                "action_low": self.action_low_np,
                "action_high": self.action_high_np,
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
        self.critic1_target.load_state_dict(checkpoint.get("critic1_target", checkpoint["critic1"]))
        self.critic2_target.load_state_dict(checkpoint.get("critic2_target", checkpoint["critic2"]))

        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic1_optimizer" in checkpoint:
            self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        if "critic2_optimizer" in checkpoint:
            self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        if (
            self.auto_alpha
            and self.alpha_optimizer is not None
            and checkpoint.get("alpha_optimizer") is not None
        ):
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(
                torch.tensor(float(checkpoint["log_alpha"]), dtype=torch.float32, device=self.device)
            )
        if "target_entropy" in checkpoint:
            self.target_entropy = float(checkpoint["target_entropy"])
        if "behavior_clone_weight" in checkpoint:
            self.behavior_clone_weight = float(checkpoint["behavior_clone_weight"])
        if "behavior_clone_q_uf_weight" in checkpoint:
            self.behavior_clone_q_uf_weight = float(checkpoint["behavior_clone_q_uf_weight"])
        if "behavior_clone_q_fp_weight" in checkpoint:
            self.behavior_clone_q_fp_weight = float(checkpoint["behavior_clone_q_fp_weight"])
        if "teacher_action_weight" in checkpoint:
            self.teacher_action_weight = float(checkpoint["teacher_action_weight"])
        if "teacher_action_q_uf_weight" in checkpoint:
            self.teacher_action_q_uf_weight = float(checkpoint["teacher_action_q_uf_weight"])
        if "teacher_action_q_fp_weight" in checkpoint:
            self.teacher_action_q_fp_weight = float(checkpoint["teacher_action_q_fp_weight"])
        if "q_fp_teacher_weight" in checkpoint:
            self.q_fp_teacher_weight = float(checkpoint["q_fp_teacher_weight"])
        if "q_fp_teacher_residual_threshold" in checkpoint:
            self.q_fp_teacher_residual_threshold = float(checkpoint["q_fp_teacher_residual_threshold"])
        if "q_fp_teacher_decision_interval" in checkpoint:
            self.q_fp_teacher_decision_interval = int(checkpoint["q_fp_teacher_decision_interval"])
        if "q_fp_teacher_q_fp_delta_max" in checkpoint:
            value = checkpoint["q_fp_teacher_q_fp_delta_max"]
            self.q_fp_teacher_q_fp_delta_max = None if value is None else float(value)
        if "q_uf_is_delta" in checkpoint:
            self.q_uf_is_delta = bool(checkpoint["q_uf_is_delta"])
        if "actor_only_update" in checkpoint:
            self.actor_only_update = bool(checkpoint["actor_only_update"])
        if "freeze_alpha_update" in checkpoint:
            self.freeze_alpha_update = bool(checkpoint["freeze_alpha_update"])
        if "distill_only_update" in checkpoint:
            self.distill_only_update = bool(checkpoint["distill_only_update"])

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
                done = bool(terminated or truncated)
                episode_reward += reward
                state = next_state

            info = info or {}
            total_rewards.append(episode_reward)
            final_masses.append(float(info.get("current_mass", info.get("final_mass", 0.0))))
            energy_costs.append(float(info.get("total_energy_cost", 0.0)))

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_final_mass": float(np.mean(final_masses)),
            "mean_energy_cost": float(np.mean(energy_costs)),
        }
