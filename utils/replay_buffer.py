"""
Replay buffer utilities.
"""

from collections import deque
from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Standard uniform replay buffer for TD3 and other off-policy baselines."""

    def __init__(self, capacity: int = 1_000_000):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        experience = Transition(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: str = "cpu"):
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        states = torch.tensor(
            np.asarray([s.state for s in samples], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            np.asarray([s.action for s in samples], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        rewards = torch.tensor(
            np.asarray([s.reward for s in samples], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.asarray([s.next_state for s in samples], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            np.asarray([s.done for s in samples], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    @property
    def size(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer used by the ESAC implementation."""

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.n_step = max(int(n_step), 1)
        self.gamma = float(gamma)
        self.frame = 1

        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _beta(self, frame_idx: int) -> float:
        return min(
            self.beta_end,
            self.beta_start + frame_idx * (self.beta_end - self.beta_start) / self.beta_frames,
        )

    def _store_experience(self, experience: Transition):
        priority = float(self.max_priority)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def _build_n_step_transition(self):
        reward = 0.0
        next_state = self.n_step_buffer[-1].next_state
        done = self.n_step_buffer[-1].done

        for idx, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * float(transition.reward)
            next_state = transition.next_state
            done = transition.done
            if done:
                break

        first = self.n_step_buffer[0]
        return Transition(
            np.asarray(first.state, dtype=np.float32),
            np.asarray(first.action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    def add(self, state, action, reward, next_state, done):
        experience = Transition(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        aggregated = self._build_n_step_transition()
        self._store_experience(aggregated)

        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                aggregated = self._build_n_step_transition()
                self._store_experience(aggregated)
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.popleft()

    def sample(self, batch_size: int, device: str = "cpu"):
        if len(self.buffer) < batch_size:
            return None

        beta = self._beta(self.frame)
        self.frame += 1

        probs = self.priorities[: len(self.buffer)]
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            probs = np.ones(len(self.buffer), dtype=np.float32) / len(self.buffer)
        else:
            probs = probs / probs_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        sample_probs = self.priorities[indices] / np.sum(self.priorities[: len(self.buffer)])
        weights = (len(self.buffer) * sample_probs) ** (-beta)
        weights = weights / weights.max()

        states = torch.tensor(np.asarray([s.state for s in samples]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.asarray([s.action for s in samples]), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.asarray([s.reward for s in samples]), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.asarray([s.next_state for s in samples]), dtype=torch.float32, device=device)
        dones = torch.tensor(np.asarray([s.done for s in samples]), dtype=torch.float32, device=device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = max(abs(float(error)), 1e-6) ** self.alpha
            self.priorities[idx] = priority
            if priority > self.max_priority:
                self.max_priority = float(priority)

    @property
    def size(self):
        return len(self.buffer)
