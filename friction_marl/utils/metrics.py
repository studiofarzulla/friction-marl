"""Metrics for friction proxies."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def compute_reward_gap(mean_reward: float, optimal_reward: float = 0.0) -> float:
    """Reward gap: optimal (Nash-equilibrium) minus realized mean reward."""
    return optimal_reward - mean_reward


def compute_convergence_time(policy_history: List[np.ndarray], delta: float = 1e-3) -> int:
    """First episode where consecutive policy vectors converge within delta."""
    for idx in range(1, len(policy_history)):
        diff = np.linalg.norm(policy_history[idx] - policy_history[idx - 1])
        if diff < delta:
            return idx
    return len(policy_history)


def compute_policy_variance(policy_vectors: List[np.ndarray]) -> float:
    """Variance of policy vectors across replications."""
    if len(policy_vectors) == 0:
        return 0.0
    stacked = np.stack(policy_vectors, axis=0)
    mean_vec = np.mean(stacked, axis=0)
    diffs = stacked - mean_vec
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def pareto_frontier(points: np.ndarray) -> np.ndarray:
    """Compute non-dominated Pareto frontier for maximization objectives."""
    frontier = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if np.all(q >= p) and np.any(q > p):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return np.array(frontier)


def compute_pareto_inefficiency(reward_vectors: np.ndarray) -> np.ndarray:
    """Distance from Pareto frontier for each reward vector."""
    if reward_vectors.size == 0:
        return np.array([])
    frontier = pareto_frontier(reward_vectors)
    distances = []
    for r in reward_vectors:
        dists = np.linalg.norm(frontier - r[None, :], axis=1)
        distances.append(float(np.min(dists)))
    return np.array(distances)
