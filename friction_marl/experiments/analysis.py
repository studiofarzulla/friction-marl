"""Analysis routines for friction proxies and regressions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from friction_marl.utils.metrics import (
    compute_pareto_inefficiency,
    compute_policy_variance,
    compute_reward_gap,
)


@dataclass
class RegressionResult:
    model: str
    target: str
    aic: float
    bic: float
    coefficients: Dict[str, float]


def _ols(y: np.ndarray, X: np.ndarray, names: List[str]) -> Tuple[Dict[str, float], float, float]:
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    sigma2 = np.mean(residuals ** 2)
    if sigma2 <= 0:
        sigma2 = 1e-12
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    coeffs = {name: float(val) for name, val in zip(names, beta)}
    return coeffs, aic, bic


def _model_matrix(df: pd.DataFrame, model: str) -> Tuple[np.ndarray, List[str]]:
    if model == "M1":
        X = np.column_stack([np.ones(len(df)), df["friction"].values])
        names = ["intercept", "friction"]
    elif model == "M2":
        additive = df["sigma"].values + df["epsilon"].values - df["alpha"].values
        X = np.column_stack([np.ones(len(df)), additive])
        names = ["intercept", "additive"]
    elif model == "M3":
        mult = df["sigma"].values * df["epsilon"].values * (1.0 - df["alpha"].values)
        X = np.column_stack([np.ones(len(df)), mult])
        names = ["intercept", "multiplicative"]
    elif model == "M4":
        X = np.column_stack(
            [np.ones(len(df)), df["alpha"].values, df["sigma"].values, df["epsilon"].values]
        )
        names = ["intercept", "alpha", "sigma", "epsilon"]
    else:
        raise ValueError(f"Unknown model: {model}")
    return X, names


def compute_metrics(replication_df: pd.DataFrame, policy_vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    group_cols = ["alpha", "sigma", "epsilon"]

    for (alpha, sigma, epsilon), group in replication_df.groupby(group_cols):
        friction = sigma * (1.0 + epsilon) / (1.0 + alpha)

        reward_gap = compute_reward_gap(float(group["mean_reward"].mean()))
        convergence_time = float(group["convergence_time"].mean())

        policy_list = []
        reward_vectors = []
        for _, row in group.iterrows():
            key = row["policy_key"]
            policy = policy_vectors.get(key)
            if policy is not None:
                policy_list.append(policy.mean(axis=0))
            agent_rewards = [row[c] for c in group.columns if c.startswith("agent_")]
            reward_vectors.append(agent_rewards)
        reward_vectors = np.array(reward_vectors, dtype=np.float32)
        pareto = compute_pareto_inefficiency(reward_vectors)

        policy_var = compute_policy_variance(policy_list)
        pareto_mean = float(np.mean(pareto)) if pareto.size else 0.0

        rows.append(
            {
                "alpha": alpha,
                "sigma": sigma,
                "epsilon": epsilon,
                "friction": friction,
                "reward_gap": reward_gap,
                "convergence_time": convergence_time,
                "policy_variance": policy_var,
                "pareto_inefficiency": pareto_mean,
            }
        )

    return pd.DataFrame(rows)


def run_regressions(metrics_df: pd.DataFrame) -> pd.DataFrame:
    models = ["M1", "M2", "M3", "M4"]
    targets = ["reward_gap", "convergence_time", "policy_variance", "pareto_inefficiency"]
    results: List[RegressionResult] = []

    for target in targets:
        y = metrics_df[target].values
        for model in models:
            X, names = _model_matrix(metrics_df, model)
            coeffs, aic, bic = _ols(y, X, names)
            results.append(
                RegressionResult(
                    model=model, target=target, aic=aic, bic=bic, coefficients=coeffs
                )
            )

    rows = []
    for res in results:
        row = {
            "model": res.model,
            "target": res.target,
            "aic": res.aic,
            "bic": res.bic,
        }
        row.update(res.coefficients)
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    replication_df = pd.read_csv(output_dir / "replication_results.csv")
    policy_path = output_dir / "policy_vectors.npz"
    policy_vectors = {}
    if policy_path.exists():
        data = np.load(policy_path, allow_pickle=True)
        policy_vectors = {k: data[k] for k in data.files}

    metrics_df = compute_metrics(replication_df, policy_vectors)
    regression_df = run_regressions(metrics_df)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(analysis_dir / "metrics.csv", index=False)
    regression_df.to_csv(analysis_dir / "regressions.csv", index=False)

    return metrics_df, regression_df
