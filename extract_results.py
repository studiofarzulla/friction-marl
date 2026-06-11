"""Extract key statistics from completed factorial experiment for paper writeup.

Generates summary_for_paper.csv with:
- Regression coefficients for H1-H4 with 95% CIs
- Model comparison (friction vs additive vs multiplicative vs independent) — AIC/BIC
- Effect sizes (eta-squared, Cohen's f)
- R-squared for each friction proxy
- Full condition means with SDs
- Convergence analysis from reward trajectories (fixing sparse policy sampling)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def compute_r_squared(y: np.ndarray, X: np.ndarray) -> float:
    """Compute R-squared from OLS."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def ols_with_ci(y: np.ndarray, X: np.ndarray, names: list[str], alpha: float = 0.05):
    """OLS with confidence intervals and p-values."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat
    dof = n - k

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof if dof > 0 else 0.0

    sigma2 = ss_res / dof if dof > 0 else 1e-12
    try:
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))

    t_crit = stats.t.ppf(1 - alpha / 2, dof) if dof > 0 else 1.96
    t_stats = beta / (se + 1e-15)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof)) if dof > 0 else np.ones_like(t_stats)

    log_lik = -0.5 * n * (np.log(2 * np.pi * max(sigma2, 1e-12)) + 1)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    results = []
    for i, name in enumerate(names):
        results.append({
            "predictor": name,
            "coefficient": float(beta[i]),
            "std_error": float(se[i]),
            "t_statistic": float(t_stats[i]),
            "p_value": float(p_values[i]),
            "ci_lower": float(beta[i] - t_crit * se[i]),
            "ci_upper": float(beta[i] + t_crit * se[i]),
        })

    return results, r_squared, adj_r_squared, aic, bic


def compute_eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    """Compute eta-squared (proportion of variance explained by grouping)."""
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0

    unique_groups = np.unique(groups)
    ss_between = 0.0
    for g in unique_groups:
        mask = groups == g
        group_mean = y[mask].mean()
        ss_between += mask.sum() * (group_mean - grand_mean) ** 2

    return ss_between / ss_total


def compute_cohens_f(eta_sq: float) -> float:
    """Convert eta-squared to Cohen's f."""
    if eta_sq >= 1.0:
        return float("inf")
    return np.sqrt(eta_sq / (1.0 - eta_sq))


def reward_based_convergence(
    learning_curves: dict[str, np.ndarray],
    replication_df: pd.DataFrame,
    window_size: int = 50,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute convergence time from reward learning curves (relative change < threshold)."""
    rows = []
    for _, rep in replication_df.iterrows():
        key = rep["policy_key"]
        lc = learning_curves.get(key)
        if lc is None:
            continue

        # Find first window where absolute relative change < threshold
        conv_window = len(lc)  # default: never converged
        for i in range(1, len(lc)):
            if abs(lc[i - 1]) > 1e-8:
                rel_change = abs((lc[i] - lc[i - 1]) / lc[i - 1])
            else:
                rel_change = abs(lc[i] - lc[i - 1])
            if rel_change < threshold:
                conv_window = i
                break

        rows.append({
            "alpha": rep["alpha"],
            "sigma": rep["sigma"],
            "epsilon": rep["epsilon"],
            "replication": rep["replication"],
            "convergence_window": conv_window,
            "convergence_episode": conv_window * window_size,
        })

    return pd.DataFrame(rows)


def main(output_dir: Path):
    print(f"Loading results from {output_dir}...")

    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)

    replication_df = pd.read_csv(output_dir / "replication_results.csv")
    print(f"Replications: {len(replication_df)}")

    # Load policy vectors
    policy_path = output_dir / "policy_vectors.npz"
    policy_vectors = {}
    if policy_path.exists():
        data = np.load(policy_path, allow_pickle=True)
        policy_vectors = {k: data[k] for k in data.files}

    # Load learning curves
    lc_path = output_dir / "learning_curves.npz"
    learning_curves = {}
    if lc_path.exists():
        data = np.load(lc_path, allow_pickle=True)
        learning_curves = {k: data[k] for k in data.files}

    # ================================================================
    # 1. Compute condition-level metrics
    # ================================================================
    print("\nComputing condition-level metrics...")

    from friction_marl.utils.metrics import compute_pareto_inefficiency, compute_policy_variance

    condition_rows = []
    group_cols = ["alpha", "sigma", "epsilon"]

    for (alpha, sigma, epsilon), group in replication_df.groupby(group_cols):
        friction = sigma * (1.0 + epsilon) / (1.0 + alpha)

        mean_reward = float(group["mean_reward"].mean())
        sd_reward = float(group["mean_reward"].std())
        reward_gap = 0.0 - mean_reward  # optimal = 0

        # Convergence from learning curves
        conv_episodes = []
        for _, row in group.iterrows():
            key = row["policy_key"]
            lc = learning_curves.get(key)
            if lc is not None:
                for i in range(1, len(lc)):
                    if abs(lc[i - 1]) > 1e-8:
                        rel_change = abs((lc[i] - lc[i - 1]) / lc[i - 1])
                    else:
                        rel_change = abs(lc[i] - lc[i - 1])
                    if rel_change < 0.01:
                        conv_episodes.append(i * 50)
                        break
                else:
                    conv_episodes.append(len(lc) * 50)

        convergence_time = float(np.mean(conv_episodes)) if conv_episodes else float("nan")

        # Policy variance across replications
        policy_list = []
        for _, row in group.iterrows():
            key = row["policy_key"]
            policy = policy_vectors.get(key)
            if policy is not None:
                policy_list.append(policy.mean(axis=0))
        policy_var = compute_policy_variance(policy_list)

        # Pareto inefficiency
        agent_cols = [c for c in group.columns if c.startswith("agent_") and c.endswith("_reward")]
        reward_vectors = group[agent_cols].values.astype(np.float32)
        pareto = compute_pareto_inefficiency(reward_vectors)
        pareto_mean = float(np.mean(pareto)) if pareto.size else 0.0

        condition_rows.append({
            "alpha": alpha,
            "sigma": sigma,
            "epsilon": epsilon,
            "friction": friction,
            "mean_reward": mean_reward,
            "sd_reward": sd_reward,
            "reward_gap": reward_gap,
            "convergence_time": convergence_time,
            "policy_variance": policy_var,
            "pareto_inefficiency": pareto_mean,
            "n_reps": len(group),
        })

    metrics_df = pd.DataFrame(condition_rows)
    metrics_df.to_csv(output_dir / "analysis" / "condition_metrics.csv", index=False)

    # ================================================================
    # 2. Regression analysis with CIs
    # ================================================================
    print("\nRunning regression analysis...")

    models = {
        "M1_friction": lambda df: (
            np.column_stack([np.ones(len(df)), df["friction"].values]),
            ["intercept", "friction"],
        ),
        "M2_additive": lambda df: (
            np.column_stack([np.ones(len(df)), df["sigma"].values + df["epsilon"].values - df["alpha"].values]),
            ["intercept", "additive"],
        ),
        "M3_multiplicative": lambda df: (
            np.column_stack([np.ones(len(df)), df["sigma"].values * df["epsilon"].values * (1.0 - df["alpha"].values)]),
            ["intercept", "multiplicative"],
        ),
        "M4_independent": lambda df: (
            np.column_stack([np.ones(len(df)), df["alpha"].values, df["sigma"].values, df["epsilon"].values]),
            ["intercept", "alpha", "sigma", "epsilon"],
        ),
    }

    targets = ["reward_gap", "convergence_time", "policy_variance", "pareto_inefficiency"]

    regression_rows = []
    model_comparison_rows = []

    for target in targets:
        y = metrics_df[target].values
        if np.all(np.isnan(y)):
            continue

        for model_name, model_fn in models.items():
            X, names = model_fn(metrics_df)
            coef_results, r2, adj_r2, aic, bic = ols_with_ci(y, X, names)

            model_comparison_rows.append({
                "target": target,
                "model": model_name,
                "r_squared": r2,
                "adj_r_squared": adj_r2,
                "aic": aic,
                "bic": bic,
            })

            for cr in coef_results:
                regression_rows.append({
                    "target": target,
                    "model": model_name,
                    **cr,
                })

    regression_df = pd.DataFrame(regression_rows)
    model_comp_df = pd.DataFrame(model_comparison_rows)

    regression_df.to_csv(output_dir / "analysis" / "regression_coefficients.csv", index=False)
    model_comp_df.to_csv(output_dir / "analysis" / "model_comparison.csv", index=False)

    # ================================================================
    # 3. Effect sizes
    # ================================================================
    print("\nComputing effect sizes...")

    effect_size_rows = []
    for factor in ["alpha", "sigma", "epsilon"]:
        for target in targets:
            y = metrics_df[target].values
            groups = metrics_df[factor].values
            eta_sq = compute_eta_squared(y, groups)
            cohens_f = compute_cohens_f(eta_sq)

            effect_size_rows.append({
                "factor": factor,
                "target": target,
                "eta_squared": eta_sq,
                "cohens_f": cohens_f,
                "effect_size": (
                    "large" if cohens_f >= 0.40 else
                    "medium" if cohens_f >= 0.25 else
                    "small" if cohens_f >= 0.10 else
                    "negligible"
                ),
            })

    effect_df = pd.DataFrame(effect_size_rows)
    effect_df.to_csv(output_dir / "analysis" / "effect_sizes.csv", index=False)

    # ================================================================
    # 4. Hypothesis tests
    # ================================================================
    print("\nTesting hypotheses H1-H4...")

    hypothesis_rows = []

    # H1: Higher friction -> larger reward gap (positive correlation)
    X_h1 = np.column_stack([np.ones(len(metrics_df)), metrics_df["friction"].values])
    coefs_h1, r2_h1, _, _, _ = ols_with_ci(metrics_df["reward_gap"].values, X_h1, ["intercept", "friction"])
    friction_coef = coefs_h1[1]
    hypothesis_rows.append({
        "hypothesis": "H1: friction -> reward_gap",
        "coefficient": friction_coef["coefficient"],
        "ci_lower": friction_coef["ci_lower"],
        "ci_upper": friction_coef["ci_upper"],
        "p_value": friction_coef["p_value"],
        "r_squared": r2_h1,
        "supported": friction_coef["coefficient"] > 0 and friction_coef["p_value"] < 0.05,
    })

    # H2: Higher friction -> slower convergence
    coefs_h2, r2_h2, _, _, _ = ols_with_ci(metrics_df["convergence_time"].values, X_h1, ["intercept", "friction"])
    friction_coef_h2 = coefs_h2[1]
    hypothesis_rows.append({
        "hypothesis": "H2: friction -> convergence_time",
        "coefficient": friction_coef_h2["coefficient"],
        "ci_lower": friction_coef_h2["ci_lower"],
        "ci_upper": friction_coef_h2["ci_upper"],
        "p_value": friction_coef_h2["p_value"],
        "r_squared": r2_h2,
        "supported": friction_coef_h2["coefficient"] > 0 and friction_coef_h2["p_value"] < 0.05,
    })

    # H3: Higher friction -> more policy variance
    coefs_h3, r2_h3, _, _, _ = ols_with_ci(metrics_df["policy_variance"].values, X_h1, ["intercept", "friction"])
    friction_coef_h3 = coefs_h3[1]
    hypothesis_rows.append({
        "hypothesis": "H3: friction -> policy_variance",
        "coefficient": friction_coef_h3["coefficient"],
        "ci_lower": friction_coef_h3["ci_lower"],
        "ci_upper": friction_coef_h3["ci_upper"],
        "p_value": friction_coef_h3["p_value"],
        "r_squared": r2_h3,
        "supported": friction_coef_h3["coefficient"] > 0 and friction_coef_h3["p_value"] < 0.05,
    })

    # H4: Higher friction -> more Pareto inefficiency
    coefs_h4, r2_h4, _, _, _ = ols_with_ci(metrics_df["pareto_inefficiency"].values, X_h1, ["intercept", "friction"])
    friction_coef_h4 = coefs_h4[1]
    hypothesis_rows.append({
        "hypothesis": "H4: friction -> pareto_inefficiency",
        "coefficient": friction_coef_h4["coefficient"],
        "ci_lower": friction_coef_h4["ci_lower"],
        "ci_upper": friction_coef_h4["ci_upper"],
        "p_value": friction_coef_h4["p_value"],
        "r_squared": r2_h4,
        "supported": friction_coef_h4["coefficient"] > 0 and friction_coef_h4["p_value"] < 0.05,
    })

    hypothesis_df = pd.DataFrame(hypothesis_rows)
    hypothesis_df.to_csv(output_dir / "analysis" / "hypothesis_tests.csv", index=False)

    # ================================================================
    # 5. Condition means with SDs
    # ================================================================
    print("\nComputing condition means...")
    condition_means = metrics_df.groupby(["alpha", "sigma", "epsilon"]).agg({
        "mean_reward": ["mean", "std"],
        "reward_gap": ["mean", "std"],
        "convergence_time": ["mean", "std"],
        "policy_variance": ["mean", "std"],
        "pareto_inefficiency": ["mean", "std"],
    }).reset_index()
    condition_means.columns = ["_".join(c).strip("_") for c in condition_means.columns]
    condition_means.to_csv(output_dir / "analysis" / "condition_means.csv", index=False)

    # ================================================================
    # Print summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    print("\n--- HYPOTHESIS TESTS ---")
    for _, h in hypothesis_df.iterrows():
        status = "SUPPORTED" if h["supported"] else "NOT SUPPORTED"
        print(f"  {h['hypothesis']}: {status}")
        print(f"    beta = {h['coefficient']:.4f} [{h['ci_lower']:.4f}, {h['ci_upper']:.4f}], p = {h['p_value']:.4e}, R^2 = {h['r_squared']:.4f}")

    print("\n--- MODEL COMPARISON ---")
    for target in targets:
        subset = model_comp_df[model_comp_df["target"] == target].sort_values("bic")
        best = subset.iloc[0]
        print(f"  {target}: best model = {best['model']} (BIC = {best['bic']:.1f}, R^2 = {best['r_squared']:.4f})")
        for _, row in subset.iterrows():
            print(f"    {row['model']}: AIC={row['aic']:.1f}, BIC={row['bic']:.1f}, R^2={row['r_squared']:.4f}")

    print("\n--- EFFECT SIZES ---")
    for _, row in effect_df.iterrows():
        print(f"  {row['factor']} -> {row['target']}: eta^2 = {row['eta_squared']:.4f}, Cohen's f = {row['cohens_f']:.4f} ({row['effect_size']})")

    print("\n--- METRICS SUMMARY ---")
    print(metrics_df[["friction", "reward_gap", "convergence_time", "policy_variance", "pareto_inefficiency"]].describe().to_string())

    # Save comprehensive summary
    summary_path = output_dir / "summary_for_paper.csv"
    all_results = pd.concat([
        hypothesis_df.assign(table="hypotheses"),
        model_comp_df.assign(table="model_comparison"),
        effect_df.assign(table="effect_sizes"),
    ], ignore_index=True)
    all_results.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(f"All analysis files in {output_dir / 'analysis'}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract results for paper")
    parser.add_argument("--output-dir", type=str, default="./results/full_factorial")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "analysis").mkdir(parents=True, exist_ok=True)
    main(output_dir)
