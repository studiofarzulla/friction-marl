"""Independent Q-Learning agent with DQN-style updates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class IQLConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay: int = 5000
    buffer_size: int = 100000
    batch_size: int = 64


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in idxs]
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.buffer)


class IQLAgent:
    """Independent Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, config: IQLConfig, seed: int | None = None):
        self.config = config
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.q = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim)
        self.target_q = QNetwork(config.obs_dim, config.action_dim, config.hidden_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=config.lr)

        self.replay = ReplayBuffer(config.buffer_size)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(self.steps / self.config.epsilon_decay, 1.0)
        return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, obs: np.ndarray) -> int:
        if self._rng.random() < self.epsilon():
            return int(self._rng.integers(0, self.action_dim))
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q(obs_t)
        return int(torch.argmax(q_vals, dim=1).item())

    def update(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)
        obs_t = torch.tensor(obs, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_vals = self.q(obs_t).gather(1, actions_t)
        with torch.no_grad():
            next_q = self.target_q(next_obs_t).max(1, keepdim=True)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def soft_update_target(self, tau: float = 0.01) -> None:
        for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def policy_vector(self, obs_batch: np.ndarray) -> np.ndarray:
        """Return average action distribution (epsilon-greedy) over a batch of observations."""
        obs_t = torch.tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            q_vals = self.q(obs_t)
        greedy_actions = torch.argmax(q_vals, dim=1).cpu().numpy()
        dist = np.zeros(self.action_dim, dtype=np.float32)
        for act in greedy_actions:
            dist[act] += 1.0
        dist /= max(len(greedy_actions), 1)
        # Blend with epsilon-greedy
        eps = self.epsilon()
        dist = (1 - eps) * dist + eps * (1.0 / self.action_dim)
        return dist
