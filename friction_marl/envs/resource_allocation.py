"""Resource allocation environment for multi-agent RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    n_agents: int = 4
    n_resources: int = 3
    capacity: float = 10.0
    episode_length: int = 100


class ResourceAllocationEnv(gym.Env):
    """Gymnasium-compatible multi-agent resource allocation environment.

    Observations are resource levels (continuous, non-negative).
    Actions per agent are vectors in {-1, 0, +1}^m.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, seed: int | None = None):
        super().__init__()
        self.config = config
        self.n_agents = config.n_agents
        self.n_resources = config.n_resources
        self.capacity = config.capacity
        self.episode_length = config.episode_length

        self.observation_space = spaces.Box(
            low=0.0,
            high=self.capacity,
            shape=(self.n_resources,),
            dtype=np.float32,
        )
        self.single_action_space = spaces.MultiDiscrete([3] * self.n_resources)
        self.action_space = spaces.Tuple(
            tuple(self.single_action_space for _ in range(self.n_agents))
        )

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._state = np.zeros(self.n_resources, dtype=np.float32)

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        self._t = 0
        self._state = self._rng.uniform(0.0, self.capacity, size=self.n_resources).astype(
            np.float32
        )
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> List[np.ndarray]:
        return [self._state.copy() for _ in range(self.n_agents)]

    def _action_to_delta(self, action: np.ndarray) -> np.ndarray:
        # Map {0,1,2} -> {-1,0,1}
        return action.astype(np.int32) - 1

    def step(self, actions: Tuple[np.ndarray, ...]):
        if len(actions) != self.n_agents:
            raise ValueError("Expected actions for all agents")

        deltas = np.zeros(self.n_resources, dtype=np.float32)
        for action in actions:
            delta = self._action_to_delta(np.asarray(action))
            deltas += delta

        # Proportional allocation: aggregate desired changes and clip to capacity
        new_state = self._state + deltas
        new_state = np.clip(new_state, 0.0, self.capacity)
        self._state = new_state.astype(np.float32)

        self._t += 1
        terminated = False
        truncated = self._t >= self.episode_length
        obs = self._get_obs()
        info = {"state": self._state.copy()}

        # Rewards are computed externally based on agent preferences
        rewards = [0.0 for _ in range(self.n_agents)]
        return obs, rewards, terminated, truncated, info

    def render(self):
        return None
