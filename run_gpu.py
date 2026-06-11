"""GPU-accelerated factorial experiment runner.

Vectorizes across all 30 replications per condition using batched parameter
tensors and torch.bmm. All computation (env, replay, Q-networks) stays on GPU
with zero CPU<->GPU transfers in the inner loop.

Compatible output format with run_parallel.py — feeds into the same analysis pipeline.

Usage:
    python run_gpu.py --output-dir ./results/gpu_factorial --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from friction_marl.experiments.factorial_design import (
    ALPHAS,
    EPSILONS,
    SIGMAS,
    Condition,
    build_action_map,
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

# ---------------------------------------------------------------------------
# Hyperparameters (match run_parallel.py defaults)
# ---------------------------------------------------------------------------
N_AGENTS = 4
N_RESOURCES = 3
HIDDEN_DIM = 64
LR = 1e-3
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY = 5000
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 64
TAU = 0.01
CAPACITY = 10.0
EPISODE_LENGTH = 100
POLICY_SAMPLE_INTERVAL = 200
LEARNING_CURVE_WINDOW = 50


# ---------------------------------------------------------------------------
# Batched Q-Network (all N agents as one set of parameter tensors)
# ---------------------------------------------------------------------------
class BatchedQNet:
    """N independent 3-layer MLPs stored as batched tensors for torch.bmm."""

    def __init__(self, n: int, obs_dim: int, act_dim: int, hidden: int, device: torch.device):
        self.n = n
        self.device = device
        # Kaiming uniform init
        k1 = (1 / obs_dim) ** 0.5
        k2 = (1 / hidden) ** 0.5
        self.w1 = torch.empty(n, obs_dim, hidden, device=device).uniform_(-k1, k1).requires_grad_(True)
        self.b1 = torch.zeros(n, 1, hidden, device=device).requires_grad_(True)
        self.w2 = torch.empty(n, hidden, hidden, device=device).uniform_(-k2, k2).requires_grad_(True)
        self.b2 = torch.zeros(n, 1, hidden, device=device).requires_grad_(True)
        self.w3 = torch.empty(n, hidden, act_dim, device=device).uniform_(-k2, k2).requires_grad_(True)
        self.b3 = torch.zeros(n, 1, act_dim, device=device).requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, batch, obs_dim) -> (N, batch, act_dim)"""
        h = torch.relu(torch.bmm(x, self.w1) + self.b1)
        h = torch.relu(torch.bmm(h, self.w2) + self.b2)
        return torch.bmm(h, self.w3) + self.b3

    def params(self) -> List[torch.Tensor]:
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def copy_from(self, src: "BatchedQNet"):
        with torch.no_grad():
            for s, d in zip(src.params(), self.params()):
                d.copy_(s)

    def soft_update_from(self, src: "BatchedQNet", tau: float = TAU):
        with torch.no_grad():
            for s, d in zip(src.params(), self.params()):
                d.mul_(1 - tau).add_(s, alpha=tau)


# ---------------------------------------------------------------------------
# GPU Replay Buffer (fully on device)
# ---------------------------------------------------------------------------
class GPUReplayBuffer:
    def __init__(self, n_agents: int, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.n = n_agents
        self.device = device
        self.obs = torch.zeros(n_agents, capacity, obs_dim, device=device)
        self.actions = torch.zeros(n_agents, capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_agents, capacity, device=device)
        self.next_obs = torch.zeros(n_agents, capacity, obs_dim, device=device)
        self.dones = torch.zeros(n_agents, capacity, device=device)
        self.ptr = torch.zeros(n_agents, dtype=torch.long, device=device)
        self.size = torch.zeros(n_agents, dtype=torch.long, device=device)
        self._arange = torch.arange(n_agents, device=device)

    def push(self, obs, actions, rewards, next_obs, dones):
        """All inputs: (N, ...) tensors on device."""
        idx = self.ptr
        self.obs[self._arange, idx] = obs
        self.actions[self._arange, idx] = actions
        self.rewards[self._arange, idx] = rewards
        self.next_obs[self._arange, idx] = next_obs
        self.dones[self._arange, idx] = dones
        self.ptr = (self.ptr + 1) % self.capacity
        self.size.clamp_(max=self.capacity - 1).add_(1).clamp_(max=self.capacity)

    def sample(self, batch_size: int):
        """Returns tuple of tensors, each (N, batch_size, ...)."""
        idx = (torch.rand(self.n, batch_size, device=self.device)
               * self.size.unsqueeze(1).float()).long()
        ar = self._arange.unsqueeze(1)
        return (
            self.obs[ar, idx],
            self.actions[ar, idx],
            self.rewards[ar, idx],
            self.next_obs[ar, idx],
            self.dones[ar, idx],
        )

    def can_sample(self, batch_size: int) -> bool:
        return bool(self.size.min().item() >= batch_size)


# ---------------------------------------------------------------------------
# Single-condition GPU runner
# ---------------------------------------------------------------------------
def run_condition(
    alpha: float,
    sigma: float,
    cond_epsilon: float,
    n_reps: int,
    n_episodes: int,
    device: torch.device,
    seed: int,
    action_map_gpu: torch.Tensor,
    probe_states_gpu: torch.Tensor,
) -> Tuple[List[dict], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run all replications for one condition on GPU.

    Returns:
        rep_rows: list of dicts (one per replication)
        policy_dict: {key: (n_agents, act_dim)} final policy vectors
        lc_dict: {key: (n_windows,)} learning curves
    """
    rng = np.random.default_rng(seed)
    N = n_reps * N_AGENTS  # 120 total independent Q-learners
    obs_dim = N_RESOURCES
    act_dim = action_map_gpu.shape[0]  # 27

    # Init networks
    q_net = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)
    target_net = BatchedQNet(N, obs_dim, act_dim, HIDDEN_DIM, device)
    target_net.copy_from(q_net)
    optimizer = torch.optim.Adam(q_net.params(), lr=LR)

    # Replay buffer
    buf = GPUReplayBuffer(N, BUFFER_CAPACITY, obs_dim, device)

    # Agent params — sample on CPU, convert to GPU tensors
    weights_np = np.zeros((n_reps, N_AGENTS, N_RESOURCES), dtype=np.float32)
    targets_np = np.zeros((n_reps, N_AGENTS, N_RESOURCES), dtype=np.float32)
    for r in range(n_reps):
        w, t = sample_agent_params(rng, N_AGENTS, N_RESOURCES, alpha, sigma)
        weights_np[r] = w
        targets_np[r] = t
    weights_t = torch.tensor(weights_np, device=device)       # (R, A, M)
    targets_t = torch.tensor(targets_np, device=device)       # (R, A, M)

    # Noise generator on GPU (pre-allocate)
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(int(rng.integers(0, 2**62)))

    # Action map for delta computation
    action_map_delta = action_map_gpu.float()  # (27, 3)

    # Probe states for policy vectors: (N, 128, obs_dim)
    probe = probe_states_gpu.unsqueeze(0).expand(N, -1, -1)

    # Tracking
    n_windows = n_episodes // LEARNING_CURVE_WINDOW
    learning_curves = torch.zeros(n_reps, n_windows, device=device)
    window_accum = torch.zeros(n_reps, device=device)
    window_count = 0

    # Policy sampling for convergence time
    policy_history = [[[] for _ in range(N_AGENTS)] for _ in range(n_reps)]

    # Last 100 episodes rewards
    last100_rewards = torch.zeros(n_reps, 100, N_AGENTS, device=device)

    global_step = 0
    _can_train = False  # avoid GPU sync every step after warmup

    for ep in range(n_episodes):
        # Reset environments: (R, M) uniform [0, CAPACITY]
        states = torch.rand(n_reps, N_RESOURCES, device=device) * CAPACITY

        ep_rewards = torch.zeros(n_reps, N_AGENTS, device=device)

        for step in range(EPISODE_LENGTH):
            global_step += 1

            # Observations: each agent sees the state -> (R, A, M) -> (N, M)
            obs_all = states.unsqueeze(1).expand(-1, N_AGENTS, -1)  # (R, A, M)

            # Add observation noise
            if cond_epsilon > 0:
                noise = torch.randn(
                    n_reps, N_AGENTS, N_RESOURCES,
                    device=device, generator=noise_gen
                ) * cond_epsilon
                obs_all = obs_all + noise

            obs_flat = obs_all.reshape(N, obs_dim)  # (N, M)

            # Epsilon-greedy
            eps = EPS_START + min(global_step / EPS_DECAY, 1.0) * (EPS_END - EPS_START)

            with torch.no_grad():
                q_vals = q_net.forward(obs_flat.unsqueeze(1)).squeeze(1)  # (N, 27)

            greedy = q_vals.argmax(dim=1)  # (N,)
            random_mask = torch.rand(N, device=device) < eps
            random_actions = torch.randint(0, act_dim, (N,), device=device)
            action_indices = torch.where(random_mask, random_actions, greedy)

            # Map indices to deltas: (N,) -> (N, M)
            deltas_flat = action_map_delta[action_indices]  # (N, M)
            deltas = deltas_flat.reshape(n_reps, N_AGENTS, N_RESOURCES)

            # Step envs
            total_delta = deltas.sum(dim=1)  # (R, M)
            new_states = (states + total_delta).clamp(0.0, CAPACITY)

            # Rewards: -sum(w * (s - t)^2, dim=-1) per agent
            diff = new_states.unsqueeze(1) - targets_t  # (R, A, M)
            utility = -(diff ** 2)
            rewards = (weights_t * utility).sum(dim=2)  # (R, A)

            # Next obs (clean, no noise — matches original env.step() behavior)
            next_obs_flat = new_states.unsqueeze(1).expand(
                -1, N_AGENTS, -1
            ).reshape(N, obs_dim)

            # Terminal
            done_val = 1.0 if step == EPISODE_LENGTH - 1 else 0.0
            dones = torch.full((N,), done_val, device=device)

            # Push to replay buffer
            buf.push(obs_flat, action_indices, rewards.reshape(N), next_obs_flat, dones)

            ep_rewards += rewards
            states = new_states

            # Q-learning update (skip GPU sync after buffer is warm)
            if not _can_train:
                _can_train = buf.can_sample(BATCH_SIZE)
            if _can_train:
                b_obs, b_act, b_rew, b_nobs, b_done = buf.sample(BATCH_SIZE)

                # Q-values: (N, BS, 27)
                q_all = q_net.forward(b_obs)
                q_taken = q_all.gather(2, b_act.unsqueeze(2)).squeeze(2)  # (N, BS)

                with torch.no_grad():
                    nq = target_net.forward(b_nobs)
                    nq_max = nq.max(dim=2)[0]  # (N, BS)
                    target_vals = b_rew + GAMMA * (1.0 - b_done) * nq_max

                # Per-agent MSE, then sum across agents (preserves gradient scale)
                loss = ((q_taken - target_vals) ** 2).mean(dim=1).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                target_net.soft_update_from(q_net, TAU)

        # --- End of episode bookkeeping ---

        # Learning curve
        mean_rew = ep_rewards.mean(dim=1) / EPISODE_LENGTH  # (R,)
        window_accum += mean_rew
        window_count += 1
        if window_count == LEARNING_CURVE_WINDOW:
            w_idx = ep // LEARNING_CURVE_WINDOW
            if w_idx < n_windows:
                learning_curves[:, w_idx] = window_accum / LEARNING_CURVE_WINDOW
            window_accum.zero_()
            window_count = 0

        # Last 100 episodes
        if ep >= n_episodes - 100:
            last100_rewards[:, ep - (n_episodes - 100)] = ep_rewards / EPISODE_LENGTH

        # Policy vectors (every POLICY_SAMPLE_INTERVAL or last episode)
        if ep % POLICY_SAMPLE_INTERVAL == 0 or ep == n_episodes - 1:
            with torch.no_grad():
                pv = q_net.forward(probe)  # (N, 128, 27)
                greedy_pv = pv.argmax(dim=2)  # (N, 128)
            greedy_pv_cpu = greedy_pv.cpu().numpy()
            for r in range(n_reps):
                for a in range(N_AGENTS):
                    idx = r * N_AGENTS + a
                    dist = np.bincount(greedy_pv_cpu[idx], minlength=act_dim).astype(np.float32)
                    dist /= 128.0
                    dist = (1 - eps) * dist + eps * (1.0 / act_dim)
                    policy_history[r][a].append(dist)

    # --- Build output ---
    rep_rows = []
    policy_dict = {}
    lc_dict = {}

    last100_cpu = last100_rewards.cpu().numpy()  # (R, 100, A)
    lc_cpu = learning_curves.cpu().numpy()       # (R, W)

    for r in range(n_reps):
        last_rew = last100_cpu[r].mean(axis=0)   # (A,)
        conv_times = []
        final_policies = []
        for a in range(N_AGENTS):
            ct = compute_convergence_time(policy_history[r][a])
            conv_times.append(ct * POLICY_SAMPLE_INTERVAL)
            final_policies.append(policy_history[r][a][-1])

        key = f"a{alpha}_s{sigma}_e{cond_epsilon}_r{r}"
        rep_rows.append({
            "alpha": alpha,
            "sigma": sigma,
            "epsilon": cond_epsilon,
            "replication": r,
            **{f"agent_{i}_reward": float(last_rew[i]) for i in range(N_AGENTS)},
            "mean_reward": float(last_rew.mean()),
            "convergence_time": float(np.mean(conv_times)),
            "policy_key": key,
        })
        policy_dict[key] = np.stack(final_policies, axis=0)
        lc_dict[key] = lc_cpu[r]

    return rep_rows, policy_dict, lc_dict


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_gpu_factorial(
    n_replications: int = 30,
    n_episodes: int = 1000,
    seed: int = 42,
    output_dir: Path = Path("./results/gpu_factorial"),
    device_id: int = 0,
):
    device = torch.device(f"cuda:{device_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    replication_path = output_dir / "replication_results.csv"
    policy_path = output_dir / "policy_vectors.npz"
    lc_path = output_dir / "learning_curves.npz"

    conditions = generate_conditions()
    action_map = build_action_map(N_RESOURCES)
    action_map_gpu = torch.tensor(action_map, device=device)

    # Probe states (shared across all conditions for comparability)
    master_rng = np.random.default_rng(seed)
    probe_states = master_rng.uniform(0.0, CAPACITY, size=(128, N_RESOURCES)).astype(np.float32)
    probe_gpu = torch.tensor(probe_states, device=device)

    # Pre-generate per-condition seeds (deterministic)
    condition_seeds = [int(master_rng.integers(0, 2**62)) for _ in conditions]

    # Load existing results for resumability
    completed = set()
    all_rep_rows: List[dict] = []
    all_policies: Dict[str, np.ndarray] = {}
    all_lcs: Dict[str, np.ndarray] = {}

    if replication_path.exists():
        existing = pd.read_csv(replication_path)
        all_rep_rows = existing.to_dict(orient="records")
        for _, row in existing.iterrows():
            completed.add((row["alpha"], row["sigma"], row["epsilon"]))
    if policy_path.exists():
        data = np.load(policy_path, allow_pickle=True)
        all_policies = {k: data[k] for k in data.files}
    if lc_path.exists():
        data = np.load(lc_path, allow_pickle=True)
        all_lcs = {k: data[k] for k in data.files}

    remaining = [(c, s) for c, s in zip(conditions, condition_seeds)
                 if (c.alpha, c.sigma, c.epsilon) not in completed]

    total = len(conditions)
    done_count = total - len(remaining)

    print(f"=== GPU Friction MARL Factorial ===")
    print(f"Device: {torch.cuda.get_device_name(device_id)}")
    print(f"Conditions: {total} (5x5x5)")
    print(f"Replications per condition: {n_replications}")
    print(f"Episodes per replication: {n_episodes}")
    print(f"Total replications: {total * n_replications}")
    print(f"Already completed: {done_count} conditions")
    print(f"Remaining: {len(remaining)} conditions")
    print()

    if not remaining:
        print("All conditions complete. Running analysis only.")
    else:
        start_time = time.time()

        for i, (cond, cseed) in enumerate(remaining):
            t0 = time.time()

            rep_rows, pol_dict, lc_dict = run_condition(
                alpha=cond.alpha,
                sigma=cond.sigma,
                cond_epsilon=cond.epsilon,
                n_reps=n_replications,
                n_episodes=n_episodes,
                device=device,
                seed=cseed,
                action_map_gpu=action_map_gpu,
                probe_states_gpu=probe_gpu,
            )

            all_rep_rows.extend(rep_rows)
            all_policies.update(pol_dict)
            all_lcs.update(lc_dict)

            elapsed_cond = time.time() - t0
            elapsed_total = time.time() - start_time
            rate = (i + 1) / elapsed_total
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

            print(
                f"  [{done_count + i + 1}/{total}] "
                f"α={cond.alpha:+.1f} σ={cond.sigma:.1f} ε={cond.epsilon:.2f}  "
                f"{elapsed_cond:.1f}s  "
                f"ETA: {eta / 60:.1f}m"
            )

            # Checkpoint every 5 conditions
            if (i + 1) % 5 == 0:
                pd.DataFrame(all_rep_rows).to_csv(replication_path, index=False)
                np.savez_compressed(policy_path, **all_policies)
                np.savez_compressed(lc_path, **all_lcs)

            # Free GPU memory between conditions
            del rep_rows, pol_dict, lc_dict
            torch.cuda.empty_cache()

        # Final save
        pd.DataFrame(all_rep_rows).to_csv(replication_path, index=False)
        np.savez_compressed(policy_path, **all_policies)
        np.savez_compressed(lc_path, **all_lcs)

        total_time = time.time() - start_time
        print(f"\nExperiment complete in {total_time / 60:.1f} minutes")
        print(f"Rate: {len(remaining) / total_time:.2f} conditions/second")

    # Generate episode-level CSV for analysis compatibility
    print("\nGenerating episode-level results...")
    episode_rows = []
    for row in all_rep_rows:
        key = row["policy_key"]
        lc = all_lcs.get(key)
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
    episode_df.to_csv(output_dir / "episode_results.csv", index=False)
    print(f"Episode results: {len(episode_rows)} rows")

    # Run analysis
    print("\nRunning analysis...")
    rep_df = pd.DataFrame(all_rep_rows)
    metrics_df, regression_df = run_analysis(output_dir)

    analysis_dir = output_dir / "analysis"
    plot_heatmaps(metrics_df, analysis_dir)
    plot_learning_curves(episode_df, analysis_dir)
    plot_regression_diagnostics(metrics_df, analysis_dir)
    plot_model_comparison(regression_df, analysis_dir)

    print(f"\n=== REGRESSION RESULTS ===")
    print(regression_df.to_string(index=False))
    print(f"\n=== METRICS SUMMARY ===")
    print(metrics_df.describe().to_string())

    return episode_df, rep_df, all_policies


def parse_args():
    p = argparse.ArgumentParser(description="GPU-accelerated friction MARL factorial")
    p.add_argument("--n-replications", type=int, default=30)
    p.add_argument("--n-episodes", type=int, default=1000)
    p.add_argument("--output-dir", type=str, default="./results/gpu_factorial")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0, help="CUDA device ID (0=7900XTX, 1=7800XT)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_gpu_factorial(
        n_replications=args.n_replications,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        device_id=args.device,
    )
