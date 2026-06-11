"""Parallel factorial experiment runner using multiprocessing.

Runs the full 5x5x5 factorial with 30 replications using all available CPU cores.
Each replication is independent and runs in its own process.

Key optimizations over original:
- Multiprocessing across all CPU cores (embarrassingly parallel)
- Single-threaded torch per worker (maximizes process parallelism)
- Policy vectors computed every 200 episodes (not every episode)
- Episode data stored as numpy binaries (not CSV) during run
- Epsilon decay bug fixed (steps counter incremented in select_action)
- Checkpointing every 30 completed replications
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
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

POLICY_SAMPLE_INTERVAL = 200  # Compute policy vector every N episodes
LEARNING_CURVE_WINDOW = 50     # Store mean reward every N episodes for learning curves


@dataclass
class JobSpec:
    """Specification for a single replication job."""
    alpha: float
    sigma: float
    epsilon: float
    rep: int
    n_agents: int
    n_resources: int
    n_episodes: int
    seed: int  # Per-replication seed (pre-computed)
    action_map: np.ndarray


def _run_one_replication(spec: JobSpec) -> Tuple[dict, np.ndarray, str, np.ndarray]:
    """Worker function: run a single replication.

    Returns: (rep_row, final_policy, policy_key, learning_curve)
    learning_curve: array of shape (n_windows,) with mean rewards per window
    """
    # Force single-threaded torch in worker to maximize process-level parallelism
    torch.set_num_threads(1)

    # Deterministic seed for this specific replication
    rng = np.random.default_rng(spec.seed)
    torch.manual_seed(spec.seed)

    cond_epsilon = spec.epsilon
    n_agents = spec.n_agents
    n_resources = spec.n_resources
    n_episodes = spec.n_episodes
    action_map = spec.action_map

    env = ResourceAllocationEnv(
        EnvConfig(n_agents=n_agents, n_resources=n_resources),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    weights, targets = sample_agent_params(
        rng, n_agents, n_resources, spec.alpha, spec.sigma
    )

    agents = []
    for _ in range(n_agents):
        cfg = IQLConfig(obs_dim=n_resources, action_dim=action_map.shape[0])
        agents.append(IQLAgent(cfg, seed=int(rng.integers(0, 2**31 - 1))))

    probe_states = rng.uniform(0.0, env.capacity, size=(128, n_resources)).astype(np.float32)

    policy_history = [[] for _ in range(n_agents)]

    # Track rewards: only store per-window means for learning curves
    n_windows = n_episodes // LEARNING_CURVE_WINDOW
    learning_curve = np.zeros(n_windows, dtype=np.float32)
    window_accumulator = 0.0
    window_count = 0

    episode_rewards_last100 = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rewards = np.zeros(n_agents, dtype=np.float32)

        while not (done or truncated):
            if cond_epsilon > 0:
                noise_all = rng.normal(0.0, cond_epsilon, size=(n_agents, n_resources))
                noisy_obs = [obs[i] + noise_all[i] for i in range(n_agents)]
            else:
                noisy_obs = obs

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

        mean_reward_ep = float(ep_rewards.mean()) / env.episode_length

        # Learning curve window accumulation
        window_accumulator += mean_reward_ep
        window_count += 1
        if window_count == LEARNING_CURVE_WINDOW:
            window_idx = ep // LEARNING_CURVE_WINDOW
            if window_idx < n_windows:
                learning_curve[window_idx] = window_accumulator / LEARNING_CURVE_WINDOW
            window_accumulator = 0.0
            window_count = 0

        # Track last 100 episodes for replication summary
        if ep >= n_episodes - 100:
            mean_rewards = ep_rewards / env.episode_length
            episode_rewards_last100.append(mean_rewards.copy())

        # Sample policy vectors periodically
        if ep % POLICY_SAMPLE_INTERVAL == 0 or ep == n_episodes - 1:
            for i, agent in enumerate(agents):
                policy_vec = agent.policy_vector(probe_states)
                policy_history[i].append(policy_vec)

    # Compute replication-level metrics
    convergence_times = [compute_convergence_time(hist) for hist in policy_history]
    convergence_times_scaled = [ct * POLICY_SAMPLE_INTERVAL for ct in convergence_times]

    final_policy = np.stack([np.array(hist[-1]) for hist in policy_history], axis=0)
    policy_key = f"a{spec.alpha}_s{spec.sigma}_e{spec.epsilon}_r{spec.rep}"

    last_rewards = np.mean(episode_rewards_last100, axis=0)
    rep_row = {
        "alpha": spec.alpha,
        "sigma": spec.sigma,
        "epsilon": spec.epsilon,
        "replication": spec.rep,
        **{f"agent_{i}_reward": float(last_rewards[i]) for i in range(n_agents)},
        "mean_reward": float(last_rewards.mean()),
        "convergence_time": float(np.mean(convergence_times_scaled)),
        "policy_key": policy_key,
    }

    return rep_row, final_policy, policy_key, learning_curve


def run_parallel_factorial(
    n_agents: int,
    n_resources: int,
    n_replications: int,
    n_episodes: int,
    seed: int,
    output_dir: Path,
    n_workers: int | None = None,
):
    if n_workers is None:
        # Each worker uses 1 torch thread, so we can use most CPU threads
        n_workers = max(1, mp.cpu_count() - 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    replication_path = output_dir / "replication_results.csv"
    policy_path = output_dir / "policy_vectors.npz"
    learning_curves_path = output_dir / "learning_curves.npz"

    conditions = generate_conditions()
    action_map = build_action_map(n_resources)

    # Load completed replications
    completed = set()
    replication_rows = []
    policy_vectors: Dict[str, np.ndarray] = {}
    learning_curves: Dict[str, np.ndarray] = {}

    if replication_path.exists():
        existing_rep = pd.read_csv(replication_path)
        replication_rows = existing_rep.to_dict(orient="records")
        for _, row in existing_rep.iterrows():
            completed.add(
                (row["alpha"], row["sigma"], row["epsilon"], int(row["replication"]))
            )
    if policy_path.exists():
        data = np.load(policy_path, allow_pickle=True)
        policy_vectors = {k: data[k] for k in data.files}
    if learning_curves_path.exists():
        data = np.load(learning_curves_path, allow_pickle=True)
        learning_curves = {k: data[k] for k in data.files}

    # Pre-compute all seeds deterministically
    master_rng = np.random.default_rng(seed)
    job_specs = []
    for cond in conditions:
        for rep in range(n_replications):
            rep_seed = int(master_rng.integers(0, 2**63 - 1))
            if (cond.alpha, cond.sigma, cond.epsilon, rep) in completed:
                continue
            job_specs.append(
                JobSpec(
                    alpha=cond.alpha,
                    sigma=cond.sigma,
                    epsilon=cond.epsilon,
                    rep=rep,
                    n_agents=n_agents,
                    n_resources=n_resources,
                    n_episodes=n_episodes,
                    seed=rep_seed,
                    action_map=action_map,
                )
            )

    total_jobs = len(conditions) * n_replications
    completed_count = total_jobs - len(job_specs)

    print(f"=== Parallel Friction MARL Factorial ===")
    print(f"Conditions: {len(conditions)} (5x5x5)")
    print(f"Replications per condition: {n_replications}")
    print(f"Episodes per replication: {n_episodes}")
    print(f"Total condition-replications: {total_jobs}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining: {len(job_specs)}")
    print(f"Workers: {n_workers}")
    print(f"Policy sample interval: {POLICY_SAMPLE_INTERVAL}")
    print(f"Learning curve window: {LEARNING_CURVE_WINDOW}")
    print()

    if len(job_specs) == 0:
        print("All replications already complete. Running analysis only.")
    else:
        start_time = time.time()
        jobs_done = 0

        # Process jobs using pool with imap_unordered for best throughput
        with mp.Pool(processes=n_workers) as pool:
            results_iter = pool.imap_unordered(_run_one_replication, job_specs, chunksize=1)

            for rep_row, final_policy, policy_key, lc in tqdm(
                results_iter, total=len(job_specs), desc="Replications"
            ):
                replication_rows.append(rep_row)
                policy_vectors[policy_key] = final_policy
                learning_curves[policy_key] = lc

                jobs_done += 1

                # Checkpoint every 30 jobs
                if jobs_done % 30 == 0:
                    rep_df = pd.DataFrame(replication_rows)
                    rep_df.to_csv(replication_path, index=False)
                    np.savez_compressed(policy_path, **policy_vectors)
                    np.savez_compressed(learning_curves_path, **learning_curves)

                    elapsed = time.time() - start_time
                    rate = jobs_done / elapsed
                    remaining_jobs = len(job_specs) - jobs_done
                    eta = remaining_jobs / rate if rate > 0 else 0
                    print(
                        f"\n  Checkpoint: {jobs_done}/{len(job_specs)} done, "
                        f"{rate:.3f} rep/s, ETA: {eta/3600:.1f}h"
                    )

        # Final save
        rep_df = pd.DataFrame(replication_rows)
        rep_df.to_csv(replication_path, index=False)
        np.savez_compressed(policy_path, **policy_vectors)
        np.savez_compressed(learning_curves_path, **learning_curves)

        total_time = time.time() - start_time
        print(f"\nExperiment complete in {total_time/3600:.2f} hours")
        print(f"Rate: {len(job_specs)/total_time:.3f} replications/second")

    # Generate episode_results.csv from learning curves (for compatibility with analysis)
    print("\nGenerating episode-level results from learning curves...")
    episode_rows = []
    for row in replication_rows:
        key = row["policy_key"]
        lc = learning_curves.get(key)
        if lc is not None:
            for w_idx, mean_r in enumerate(lc):
                episode_rows.append({
                    "alpha": row["alpha"],
                    "sigma": row["sigma"],
                    "epsilon": row["epsilon"],
                    "replication": row["replication"],
                    "episode": w_idx * LEARNING_CURVE_WINDOW + LEARNING_CURVE_WINDOW // 2,
                    "mean_reward": float(mean_r),
                })
    episode_df = pd.DataFrame(episode_rows)
    episode_path = output_dir / "episode_results.csv"
    episode_df.to_csv(episode_path, index=False)
    print(f"Episode results: {len(episode_rows)} rows written")

    # Run analysis
    print("\nRunning analysis...")
    metrics_df, regression_df = run_analysis(output_dir)

    analysis_dir = output_dir / "analysis"
    plot_heatmaps(metrics_df, analysis_dir)
    plot_learning_curves(episode_df, analysis_dir)
    plot_regression_diagnostics(metrics_df, analysis_dir)
    plot_model_comparison(regression_df, analysis_dir)
    print(f"Analysis complete. Plots saved to {analysis_dir}")

    # Print summary statistics
    print("\n=== REGRESSION RESULTS ===")
    print(regression_df.to_string(index=False))

    print("\n=== METRICS SUMMARY ===")
    print(metrics_df.describe().to_string())

    return episode_df, rep_df, policy_vectors


def parse_args():
    parser = argparse.ArgumentParser(description="Run parallel friction MARL factorial")
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--n-resources", type=int, default=3)
    parser.add_argument("--n-replications", type=int, default=30)
    parser.add_argument("--n-episodes", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="./results/full_factorial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 2)")
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    run_parallel_factorial(
        n_agents=args.n_agents,
        n_resources=args.n_resources,
        n_replications=args.n_replications,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        n_workers=args.workers,
    )
