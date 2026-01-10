"""Visualization utilities for analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")


def plot_heatmaps(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["reward_gap", "convergence_time", "policy_variance", "pareto_inefficiency"]:
        pivot = metrics_df.pivot_table(index="alpha", columns="sigma", values=metric, aggfunc="mean")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"{metric} across alpha x sigma")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{metric}_alpha_sigma.png")
        plt.close()

        pivot = metrics_df.pivot_table(index="alpha", columns="epsilon", values=metric, aggfunc="mean")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma")
        plt.title(f"{metric} across alpha x epsilon")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_{metric}_alpha_epsilon.png")
        plt.close()


def plot_learning_curves(episode_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = episode_df.groupby("episode")["mean_reward"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    plt.plot(curve["episode"], curve["mean_reward"], color="steelblue")
    plt.title("Average learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve_mean_reward.png")
    plt.close()


def plot_regression_diagnostics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["reward_gap", "convergence_time", "policy_variance", "pareto_inefficiency"]:
        x = metrics_df["friction"].values
        y = metrics_df[metric].values
        coeffs = np.polyfit(x, y, deg=1)
        fitted = coeffs[0] * x + coeffs[1]
        residuals = y - fitted

        plt.figure(figsize=(6, 5))
        plt.scatter(fitted, residuals, alpha=0.7)
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
        plt.title(f"Residuals vs fitted ({metric})")
        plt.xlabel("Fitted")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.savefig(output_dir / f"residuals_{metric}.png")
        plt.close()


def plot_model_comparison(regression_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in regression_df["target"].unique():
        subset = regression_df[regression_df["target"] == metric]
        plt.figure(figsize=(7, 4))
        sns.barplot(data=subset, x="model", y="bic", palette="deep")
        plt.title(f"Model comparison (BIC) - {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_bic_{metric}.png")
        plt.close()

        plt.figure(figsize=(7, 4))
        sns.barplot(data=subset, x="model", y="aic", palette="muted")
        plt.title(f"Model comparison (AIC) - {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_aic_{metric}.png")
        plt.close()
