"""Factorial design experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from friction_marl.agents.iql import IQLAgent, IQLConfig
from friction_marl.envs.resource_allocation import EnvConfig, ResourceAllocationEnv
from friction_marl.utils.metrics import compute_convergence_time


@dataclass
class Condition:
    alpha: float
    sigma: float
    epsilon: float


ALPHAS = [-0.8, -0.4, 0.0, 0.4, 0.8]
SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
EPSILONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def generate_conditions() -> List[Condition]:
    return [Condition(a, s, e) for a in ALPHAS for s in SIGMAS for e in EPSILONS]


def sample_agent_params(
    rng: np.random.Generator,
    n_agents: int,
    n_resources: int,
    alpha: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample weights and target preferences with alignment control."""
    base_target = rng.normal(0.0, 1.0, size=n_resources)
    targets = []
    for _ in range(n_agents):
        noise = rng.normal(0.0, 1.0, size=n_resources)
        target = alpha * base_target + (1.0 - abs(alpha)) * noise
        targets.append(target)
    targets_arr = np.stack(targets, axis=0)

    weights = rng.normal(loc=sigma, scale=0.05, size=(n_agents, n_resources))
    weights = np.clip(weights, 0.0, 1.0)
    return weights, targets_arr


def build_action_map(n_resources: int) -> np.ndarray:
    """Map action index -> action vector in {-1,0,1}^m."""
    values = [-1, 0, 1]
    grid = np.array(np.meshgrid(*([values] * n_resources), indexing="ij"))
    actions = grid.reshape(n_resources, -1).T
    return actions


def compute_rewards(
    state: np.ndarray, weights: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute agent rewards given current state."""
    diff = state[None, :] - targets
    utility = -(diff ** 2)
    return np.sum(weights * utility, axis=1)


def run_factorial_experiment(
    n_agents: int,
    n_resources: int,
    n_replications: int,
    n_episodes: int,
    seed: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    conditions = generate_conditions()
    action_map = build_action_map(n_resources)

    episode_rows: List[dict] = []
    replication_rows: List[dict] = []
    policy_vectors: Dict[str, np.ndarray] = {}

    for cond in tqdm(conditions, desc="Conditions"):
        for rep in range(n_replications):
            env_seed = int(rng.integers(0, 2**31 - 1))
            env = ResourceAllocationEnv(
                EnvConfig(n_agents=n_agents, n_resources=n_resources), seed=env_seed
            )
            weights, targets = sample_agent_params(
                rng, n_agents, n_resources, cond.alpha, cond.sigma
            )

            agents = []
            for agent_idx in range(n_agents):
                cfg = IQLConfig(
                    obs_dim=n_resources,
                    action_dim=action_map.shape[0],
                )
                agents.append(IQLAgent(cfg, seed=int(rng.integers(0, 2**31 - 1))))

            obs, _ = env.reset(seed=env_seed)
            probe_states = rng.uniform(0.0, env.capacity, size=(128, n_resources)).astype(
                np.float32
            )

            policy_history = [[] for _ in range(n_agents)]
            episode_rewards = []

            for ep in range(n_episodes):
                obs, _ = env.reset()
                done = False
                truncated = False
                ep_rewards = np.zeros(n_agents, dtype=np.float32)

                while not (done or truncated):
                    noisy_obs = []
                    for agent_obs in obs:
                        noise = rng.normal(0.0, cond.epsilon, size=n_resources)
                        noisy_obs.append(agent_obs + noise)

                    action_indices = [agent.select_action(o) for agent, o in zip(agents, noisy_obs)]
                    actions = [action_map[idx] for idx in action_indices]

                    next_obs, _, done, truncated, info = env.step(tuple(actions))
                    state = info["state"]
                    rewards = compute_rewards(state, weights, targets)

                    for i, agent in enumerate(agents):
                        agent.replay.push(noisy_obs[i], action_indices[i], rewards[i], next_obs[i], done or truncated)
                        agent.update()
                        agent.soft_update_target()

                    ep_rewards += rewards
                    obs = next_obs

                mean_rewards = ep_rewards / env.episode_length
                episode_rewards.append(mean_rewards)

                for i, agent in enumerate(agents):
                    policy_vec = agent.policy_vector(probe_states)
                    policy_history[i].append(policy_vec)

                episode_rows.append(
                    {
                        "alpha": cond.alpha,
                        "sigma": cond.sigma,
                        "epsilon": cond.epsilon,
                        "replication": rep,
                        "episode": ep,
                        **{f"agent_{i}_reward": mean_rewards[i] for i in range(n_agents)},
                        "mean_reward": float(mean_rewards.mean()),
                    }
                )

            convergence_times = [
                compute_convergence_time(hist) for hist in policy_history
            ]
            final_policy = np.stack([np.array(hist[-1]) for hist in policy_history], axis=0)
            policy_key = f"a{cond.alpha}_s{cond.sigma}_e{cond.epsilon}_r{rep}"
            policy_vectors[policy_key] = final_policy

            last_rewards = np.mean(episode_rewards[-100:], axis=0)
            replication_rows.append(
                {
                    "alpha": cond.alpha,
                    "sigma": cond.sigma,
                    "epsilon": cond.epsilon,
                    "replication": rep,
                    **{f"agent_{i}_reward": float(last_rewards[i]) for i in range(n_agents)},
                    "mean_reward": float(last_rewards.mean()),
                    "convergence_time": float(np.mean(convergence_times)),
                    "policy_key": policy_key,
                }
            )

    episode_df = pd.DataFrame(episode_rows)
    replication_df = pd.DataFrame(replication_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_df.to_csv(output_dir / "episode_results.csv", index=False)
    replication_df.to_csv(output_dir / "replication_results.csv", index=False)

    return episode_df, replication_df, policy_vectors
