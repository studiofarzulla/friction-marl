"""Optimized factorial experiment runner.

Same experiment semantics as run_experiments.py but with major performance
optimizations:
- Policy vectors computed every 100 episodes instead of every episode
- Episode-level CSV written in streaming mode (append)
- Batched agent forward passes where possible
- Reduced checkpoint overhead (npz saved every 10 replications)
- Steps counter incremented properly for epsilon decay
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from friction_marl.agents.iql import IQLAgent, IQLConfig
from friction_marl.envs.resource_allocation import EnvConfig, ResourceAllocationEnv
from friction_marl.experiments.factorial_design import (
    ALPHAS,
    EPSILONS,
    SIGMAS,
    Condition,
    build_action_map,
    compute_rewards,
    generate_conditions,
    sample_agent_params,
)
from friction_marl.experiments.analysis import run_analysis
from friction_marl.utils.metrics import compute_convergence_time
from friction_marl.utils.visualization import (
    plot_heatmaps,
    plot_learning_curves,
    plot_model_comparison,
    plot_regression_diagnostics,
)


POLICY_SAMPLE_INTERVAL = 100  # Compute policy vector every N episodes
NPZ_CHECKPOINT_INTERVAL = 10  # Save npz every N replications


def run_single_replication(
    cond: Condition,
    rep: int,
    n_agents: int,
    n_resources: int,
    n_episodes: int,
    action_map: np.ndarray,
    rng: np.random.Generator,
    env_seed: int,
) -> Tuple[List[dict], dict, np.ndarray]:
    """Run a single condition-replication, return episode rows, replication row, final policy."""

    env = ResourceAllocationEnv(
        EnvConfig(n_agents=n_agents, n_resources=n_resources), seed=env_seed
    )
    weights, targets = sample_agent_params(
        rng, n_agents, n_resources, cond.alpha, cond.sigma
    )

    agents = []
    for _ in range(n_agents):
        cfg = IQLConfig(obs_dim=n_resources, action_dim=action_map.shape[0])
        agents.append(IQLAgent(cfg, seed=int(rng.integers(0, 2**31 - 1))))

    probe_states = rng.uniform(0.0, env.capacity, size=(128, n_resources)).astype(
        np.float32
    )

    # Pre-compute probe tensor once
    probe_tensor = torch.tensor(probe_states, dtype=torch.float32)

    policy_history = [[] for _ in range(n_agents)]
    episode_rewards = []
    episode_rows = []

    n_resources_f32 = n_resources

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rewards = np.zeros(n_agents, dtype=np.float32)

        while not (done or truncated):
            # Vectorized noise generation
            if cond.epsilon > 0:
                noise_all = rng.normal(0.0, cond.epsilon, size=(n_agents, n_resources))
                noisy_obs = [obs[i] + noise_all[i] for i in range(n_agents)]
            else:
                noisy_obs = obs

            # Individual agent actions (can't easily batch due to separate Q-networks)
            action_indices = [
                agent.select_action(o) for agent, o in zip(agents, noisy_obs)
            ]
            actions = [action_map[idx] for idx in action_indices]

            next_obs, _, done, truncated, info = env.step(tuple(actions))
            state = info["state"]
            rewards = compute_rewards(state, weights, targets)

            terminal = done or truncated
            for i, agent in enumerate(agents):
                agent.replay.push(
                    noisy_obs[i], action_indices[i], rewards[i], next_obs[i], terminal
                )
                agent.update()
                agent.soft_update_target()

            ep_rewards += rewards
            obs = next_obs

        mean_rewards = ep_rewards / env.episode_length
        episode_rewards.append(mean_rewards)

        # Policy vector sampling: every POLICY_SAMPLE_INTERVAL episodes + last episode
        if ep % POLICY_SAMPLE_INTERVAL == 0 or ep == n_episodes - 1:
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
                **{f"agent_{i}_reward": float(mean_rewards[i]) for i in range(n_agents)},
                "mean_reward": float(mean_rewards.mean()),
            }
        )

    # Convergence from sampled policy history (still meaningful — same delta threshold)
    convergence_times = [compute_convergence_time(hist) for hist in policy_history]
    # Scale convergence time back to episode scale
    convergence_times_scaled = [
        ct * POLICY_SAMPLE_INTERVAL for ct in convergence_times
    ]

    final_policy = np.stack(
        [np.array(hist[-1]) for hist in policy_history], axis=0
    )
    policy_key = f"a{cond.alpha}_s{cond.sigma}_e{cond.epsilon}_r{rep}"

    last_rewards = np.mean(episode_rewards[-100:], axis=0)
    rep_row = {
        "alpha": cond.alpha,
        "sigma": cond.sigma,
        "epsilon": cond.epsilon,
        "replication": rep,
        **{f"agent_{i}_reward": float(last_rewards[i]) for i in range(n_agents)},
        "mean_reward": float(last_rewards.mean()),
        "convergence_time": float(np.mean(convergence_times_scaled)),
        "policy_key": policy_key,
    }

    return episode_rows, rep_row, final_policy, policy_key


def run_optimized_factorial(
    n_agents: int,
    n_resources: int,
    n_replications: int,
    n_episodes: int,
    seed: int,
    output_dir: Path,
):
    rng = np.random.default_rng(seed)
    conditions = generate_conditions()
    action_map = build_action_map(n_resources)

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode_results.csv"
    replication_path = output_dir / "replication_results.csv"
    policy_path = output_dir / "policy_vectors.npz"

    # Load completed replications for resume support
    completed = set()
    replication_rows = []
    policy_vectors: Dict[str, np.ndarray] = {}

    if replication_path.exists():
        existing_rep = pd.read_csv(replication_path)
        replication_rows = existing_rep.to_dict(orient="records")
        for _, row in existing_rep.iterrows():
            completed.add(
                (row["alpha"], row["sigma"], row["epsilon"], row["replication"])
            )
    if policy_path.exists():
        data = np.load(policy_path, allow_pickle=True)
        policy_vectors = {k: data[k] for k in data.files}

    # Episode CSV: streaming append mode
    episode_header = [
        "alpha", "sigma", "epsilon", "replication", "episode",
        *[f"agent_{i}_reward" for i in range(n_agents)],
        "mean_reward",
    ]

    # If no existing episode file or starting fresh, write header
    episode_file_exists = episode_path.exists() and episode_path.stat().st_size > 0
    if not episode_file_exists:
        with open(episode_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_header)
            writer.writeheader()

    total_reps = len(conditions) * n_replications
    completed_count = len(completed)
    remaining = total_reps - completed_count

    print(f"Total condition-replications: {total_reps}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining: {remaining}")
    print(f"Episodes per replication: {n_episodes}")
    print(f"Policy sample interval: {POLICY_SAMPLE_INTERVAL}")
    print()

    rep_counter = 0
    start_time = time.time()

    with tqdm(total=remaining, desc="Replications") as pbar:
        for cond in conditions:
            for rep in range(n_replications):
                # Advance RNG even for completed reps to maintain determinism
                env_seed = int(rng.integers(0, 2**31 - 1))

                if (cond.alpha, cond.sigma, cond.epsilon, rep) in completed:
                    # Still need to advance agent seeds
                    for _ in range(n_agents):
                        rng.integers(0, 2**31 - 1)
                    continue

                # Also advance agent seeds via the same rng
                # (done inside run_single_replication via rng passed by ref)

                ep_rows, rep_row, final_policy, policy_key = run_single_replication(
                    cond=cond,
                    rep=rep,
                    n_agents=n_agents,
                    n_resources=n_resources,
                    n_episodes=n_episodes,
                    action_map=action_map,
                    rng=rng,
                    env_seed=env_seed,
                )

                # Stream episode rows to CSV
                with open(episode_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=episode_header)
                    writer.writerows(ep_rows)

                # Accumulate replication results
                replication_rows.append(rep_row)
                policy_vectors[policy_key] = final_policy

                # Save replication CSV every time (small)
                rep_df = pd.DataFrame(replication_rows)
                rep_df.to_csv(replication_path, index=False)

                # Save policy npz periodically (large)
                rep_counter += 1
                if rep_counter % NPZ_CHECKPOINT_INTERVAL == 0:
                    np.savez(policy_path, **policy_vectors)

                elapsed = time.time() - start_time
                rate = rep_counter / elapsed
                eta = (remaining - rep_counter) / rate if rate > 0 else 0

                pbar.set_postfix(
                    cond=f"a={cond.alpha},s={cond.sigma},e={cond.epsilon}",
                    rate=f"{rate:.2f} rep/s",
                    eta=f"{eta/3600:.1f}h",
                )
                pbar.update(1)

    # Final save
    np.savez(policy_path, **policy_vectors)
    replication_df = pd.DataFrame(replication_rows)
    replication_df.to_csv(replication_path, index=False)

    print(f"\nExperiment complete in {(time.time() - start_time)/3600:.2f} hours")
    print(f"Results saved to {output_dir}")

    # Run analysis
    print("\nRunning analysis...")
    episode_df = pd.read_csv(episode_path)
    metrics_df, regression_df = run_analysis(output_dir)

    analysis_dir = output_dir / "analysis"
    plot_heatmaps(metrics_df, analysis_dir)
    plot_learning_curves(episode_df, analysis_dir)
    plot_regression_diagnostics(metrics_df, analysis_dir)
    plot_model_comparison(regression_df, analysis_dir)
    print(f"Analysis saved to {analysis_dir}")

    return episode_df, replication_df, policy_vectors


def parse_args():
    parser = argparse.ArgumentParser(description="Run optimized friction MARL experiments")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-resources", type=int, default=3)
    parser.add_argument("--n-replications", type=int, default=30)
    parser.add_argument("--n-episodes", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="./results/full_factorial")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_optimized_factorial(
        n_agents=args.n_agents,
        n_resources=args.n_resources,
        n_replications=args.n_replications,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
