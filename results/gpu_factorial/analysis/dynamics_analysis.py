"""
Learning dynamics & policy convergence analysis for the MARL 5x5x5 factorial.

Generates:
  1. Learning curve comparison plots (alpha, by noise level)
  2. Convergence heatmap (alpha x sigma)
  3. Policy space PCA/t-SNE visualizations
  4. Agent specialization analysis
  5. Reward dynamics for key conditions

All figures saved as 300 DPI PNG + PDF.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path("/home/purrpower/Resurrexi/projects/papers/github-repos/friction-marl/results/gpu_factorial")
OUT = BASE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Colorblind-friendly palette (Tol's bright)
ALPHA_COLORS = {
    -0.8: "#332288",  # indigo
    -0.4: "#88CCEE",  # cyan
     0.0: "#999999",  # grey
     0.4: "#EE6677",  # rose
     0.8: "#CC3311",  # red
}
ALPHA_LABELS = {
    -0.8: r"$\alpha=-0.8$ (cooperative)",
    -0.4: r"$\alpha=-0.4$",
     0.0: r"$\alpha=0.0$ (neutral)",
     0.4: r"$\alpha=+0.4$",
     0.8: r"$\alpha=+0.8$ (adversarial)",
}

# Hyperparams from run_gpu.py
N_EPISODES = 1000
LC_WINDOW = 50
N_WINDOWS = N_EPISODES // LC_WINDOW  # 20
EPISODE_TICKS = np.arange(N_WINDOWS) * LC_WINDOW + LC_WINDOW // 2

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def savefig(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}.png + .pdf")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(BASE / "replication_results.csv")
lc_data = np.load(BASE / "learning_curves.npz")
pv_data = np.load(BASE / "policy_vectors.npz")
print(f"  {len(df)} replications, {len(lc_data.files)} learning curves, {len(pv_data.files)} policy vectors")


def get_condition_lcs(alpha, sigma, epsilon):
    """Get all 30 learning curves for a condition. Returns (30, 20) array."""
    curves = []
    for r in range(30):
        key = f"a{alpha}_s{sigma}_e{epsilon}_r{r}"
        if key in lc_data:
            curves.append(lc_data[key])
    return np.array(curves) if curves else None


# =========================================================================
# 1. LEARNING CURVE ANALYSIS
# =========================================================================
print("\n=== 1. Learning Curve Analysis ===")

# --- Fig 1a: Alpha comparison at sigma=0.6, epsilon=0.5 ---
fig, ax = plt.subplots(figsize=(8, 5))
for alpha in [-0.8, 0.0, 0.8]:
    curves = get_condition_lcs(alpha, 0.6, 0.5)
    if curves is not None:
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(EPISODE_TICKS, mean, color=ALPHA_COLORS[alpha], label=ALPHA_LABELS[alpha], linewidth=2)
        ax.fill_between(EPISODE_TICKS, mean - std, mean + std, color=ALPHA_COLORS[alpha], alpha=0.15)

ax.set_xlabel("Episode")
ax.set_ylabel("Mean Reward (per step)")
ax.set_title(r"Learning Curves: $\alpha$ Comparison ($\sigma=0.6$, $\varepsilon=0.5$)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
savefig(fig, "fig1a_learning_curves_alpha_comparison")

# --- Fig 1b: Alpha comparison at low noise (sigma=0.2) ---
fig, ax = plt.subplots(figsize=(8, 5))
for alpha in [-0.8, -0.4, 0.0, 0.4, 0.8]:
    curves = get_condition_lcs(alpha, 0.2, 0.5)
    if curves is not None:
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(EPISODE_TICKS, mean, color=ALPHA_COLORS[alpha], label=ALPHA_LABELS[alpha], linewidth=1.8)
        ax.fill_between(EPISODE_TICKS, mean - std, mean + std, color=ALPHA_COLORS[alpha], alpha=0.1)

ax.set_xlabel("Episode")
ax.set_ylabel("Mean Reward (per step)")
ax.set_title(r"Learning Curves at Low Noise ($\sigma=0.2$, $\varepsilon=0.5$)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
savefig(fig, "fig1b_learning_curves_low_noise")

# --- Fig 1c: Alpha comparison at high noise (sigma=1.0) ---
fig, ax = plt.subplots(figsize=(8, 5))
for alpha in [-0.8, -0.4, 0.0, 0.4, 0.8]:
    curves = get_condition_lcs(alpha, 1.0, 0.5)
    if curves is not None:
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(EPISODE_TICKS, mean, color=ALPHA_COLORS[alpha], label=ALPHA_LABELS[alpha], linewidth=1.8)
        ax.fill_between(EPISODE_TICKS, mean - std, mean + std, color=ALPHA_COLORS[alpha], alpha=0.1)

ax.set_xlabel("Episode")
ax.set_ylabel("Mean Reward (per step)")
ax.set_title(r"Learning Curves at High Noise ($\sigma=1.0$, $\varepsilon=0.5$)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
savefig(fig, "fig1c_learning_curves_high_noise")

# --- Fig 1d: 3x5 panel — sigma rows, alpha columns ---
fig, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey='row')
sigma_levels = [0.2, 0.6, 1.0]
alpha_levels = [-0.8, -0.4, 0.0, 0.4, 0.8]

for i, sigma in enumerate(sigma_levels):
    for j, alpha in enumerate(alpha_levels):
        ax = axes[i, j]
        for eps in [0.0, 0.5, 1.0]:
            curves = get_condition_lcs(alpha, sigma, eps)
            if curves is not None:
                mean = curves.mean(axis=0)
                label = f"$\\varepsilon={eps:.1f}$" if i == 0 and j == 0 else None
                style = ['-', '--', ':'][[0.0, 0.5, 1.0].index(eps)]
                ax.plot(EPISODE_TICKS, mean, linestyle=style, linewidth=1.2,
                        color=ALPHA_COLORS[alpha], label=label)
        if i == 0:
            ax.set_title(ALPHA_LABELS[alpha], fontsize=8)
        if j == 0:
            ax.set_ylabel(f"$\\sigma={sigma}$\nMean Reward", fontsize=8)
        if i == 2:
            ax.set_xlabel("Episode", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

axes[0, 0].legend(fontsize=7, loc='lower right')
fig.suptitle("Learning Curves: Full Factorial Grid", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "fig1d_learning_curves_full_grid")

# --- Learning speed: episodes to 80% of final performance ---
print("\n  Computing learning speed (80% threshold)...")
speed_rows = []
for _, row in df.drop_duplicates(subset=['alpha','sigma','epsilon']).iterrows():
    alpha, sigma, eps = row['alpha'], row['sigma'], row['epsilon']
    curves = get_condition_lcs(alpha, sigma, eps)
    if curves is None:
        continue
    final_perf = curves[:, -3:].mean(axis=1)  # average of last 3 windows
    threshold = curves[:, 0] + 0.8 * (final_perf - curves[:, 0])  # 80% of improvement

    episodes_to_80 = []
    for r in range(len(curves)):
        reached = np.where(curves[r] >= threshold[r])[0]
        if len(reached) > 0:
            episodes_to_80.append(EPISODE_TICKS[reached[0]])
        else:
            episodes_to_80.append(N_EPISODES)  # never reached

    speed_rows.append({
        'alpha': alpha, 'sigma': sigma, 'epsilon': eps,
        'mean_episodes_to_80': np.mean(episodes_to_80),
        'median_episodes_to_80': np.median(episodes_to_80),
    })

speed_df = pd.DataFrame(speed_rows)

# Fig 1e: Learning speed heatmap (alpha x sigma, averaged over epsilon)
pivot = speed_df.groupby(['alpha','sigma'])['mean_episodes_to_80'].mean().reset_index()
pivot_table = pivot.pivot(index='alpha', columns='sigma', values='mean_episodes_to_80')

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            cbar_kws={'label': 'Episodes to 80% Performance'})
ax.set_title("Learning Speed: Episodes to Reach 80% of Final Performance")
ax.set_xlabel(r"$\sigma$ (preference spread)")
ax.set_ylabel(r"$\alpha$ (friction)")
savefig(fig, "fig1e_learning_speed_heatmap")

# --- Learning stability: variance in last 100 episodes ---
print("  Computing learning stability...")
stability_rows = []
for _, row in df.iterrows():
    key = row['policy_key']
    if key in lc_data:
        curve = lc_data[key]
        # Stability = std of last 5 windows (last 250 episodes)
        stability_rows.append({
            'alpha': row['alpha'], 'sigma': row['sigma'], 'epsilon': row['epsilon'],
            'late_std': curve[-5:].std(),
            'late_mean': curve[-5:].mean(),
        })

stability_df = pd.DataFrame(stability_rows)

# Fig 1f: Stability (late variance) by alpha and sigma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# By alpha
stab_alpha = stability_df.groupby('alpha')['late_std'].agg(['mean','std']).reset_index()
ax1.bar(range(5), stab_alpha['mean'], yerr=stab_alpha['std'], capsize=4,
        color=[ALPHA_COLORS[a] for a in stab_alpha['alpha']])
ax1.set_xticks(range(5))
ax1.set_xticklabels([f"{a:.1f}" for a in stab_alpha['alpha']])
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel("Std of Reward (Last 250 Episodes)")
ax1.set_title("Learning Stability by Friction Level")
ax1.grid(True, alpha=0.3)

# By sigma
stab_sigma = stability_df.groupby('sigma')['late_std'].agg(['mean','std']).reset_index()
ax2.bar(range(5), stab_sigma['mean'], yerr=stab_sigma['std'], capsize=4,
        color=sns.color_palette("crest", 5))
ax2.set_xticks(range(5))
ax2.set_xticklabels([f"{s:.1f}" for s in stab_sigma['sigma']])
ax2.set_xlabel(r"$\sigma$")
ax2.set_ylabel("Std of Reward (Last 250 Episodes)")
ax2.set_title("Learning Stability by Preference Spread")
ax2.grid(True, alpha=0.3)

fig.suptitle("Learning Stability Analysis", fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, "fig1f_learning_stability")


# =========================================================================
# 2. CONVERGENCE ANALYSIS
# =========================================================================
print("\n=== 2. Convergence Analysis ===")

MAX_CONV = 1200.0  # max convergence time in our data

# Convergence rate: fraction with convergence_time < MAX_CONV
df['converged'] = df['convergence_time'] < MAX_CONV

# --- Fig 2a: Convergence heatmap alpha x sigma ---
conv_rate = df.groupby(['alpha','sigma'])['converged'].mean().reset_index()
conv_pivot = conv_rate.pivot(index='alpha', columns='sigma', values='converged')

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(conv_pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
            vmin=0, vmax=0.1, cbar_kws={'label': 'Convergence Rate'})
ax.set_title(r"Policy Convergence Rate ($\alpha \times \sigma$, averaged over $\varepsilon$)")
ax.set_xlabel(r"$\sigma$ (preference spread)")
ax.set_ylabel(r"$\alpha$ (friction)")
savefig(fig, "fig2a_convergence_heatmap")

# --- Fig 2b: Convergence by all three factors ---
conv_full = df.groupby(['alpha','sigma','epsilon'])['converged'].mean().reset_index()

fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
for i, eps in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    ax = axes[i]
    sub = conv_full[conv_full['epsilon'] == eps]
    piv = sub.pivot(index='alpha', columns='sigma', values='converged')
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=0.15, cbar=i == 4)
    ax.set_title(f"$\\varepsilon={eps}$")
    if i > 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel(r"$\alpha$")
    ax.set_xlabel(r"$\sigma$")

fig.suptitle("Policy Convergence Rate by Condition", fontsize=13, y=1.03)
fig.tight_layout()
savefig(fig, "fig2b_convergence_by_epsilon")

# --- Convergence analysis: using reward-based convergence instead ---
# Since policy convergence is very rare, let's also measure reward plateau
print("  Computing reward-based convergence (plateau detection)...")
reward_conv_rows = []
for _, row in df.iterrows():
    key = row['policy_key']
    if key in lc_data:
        curve = lc_data[key]
        # Reward converged = first window where subsequent windows stay within 10% of final
        final_val = curve[-3:].mean()
        if abs(final_val) < 1e-8:
            reward_conv_rows.append({
                'alpha': row['alpha'], 'sigma': row['sigma'], 'epsilon': row['epsilon'],
                'reward_converged': True, 'reward_conv_episode': 25
            })
            continue

        tolerance = 0.1 * abs(final_val)
        conv_ep = N_EPISODES  # default: never
        for w in range(len(curve)):
            if all(abs(curve[w2] - final_val) <= tolerance for w2 in range(w, len(curve))):
                conv_ep = EPISODE_TICKS[w]
                break

        reward_conv_rows.append({
            'alpha': row['alpha'], 'sigma': row['sigma'], 'epsilon': row['epsilon'],
            'reward_converged': conv_ep < N_EPISODES,
            'reward_conv_episode': conv_ep,
        })

rconv_df = pd.DataFrame(reward_conv_rows)

# Fig 2c: Reward convergence rate heatmap
rconv_rate = rconv_df.groupby(['alpha','sigma'])['reward_converged'].mean().reset_index()
rconv_pivot = rconv_rate.pivot(index='alpha', columns='sigma', values='reward_converged')

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(rconv_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
            vmin=0, vmax=1.0, cbar_kws={'label': 'Reward Convergence Rate'})
ax.set_title(r"Reward Convergence Rate (10% tolerance of final)")
ax.set_xlabel(r"$\sigma$ (preference spread)")
ax.set_ylabel(r"$\alpha$ (friction)")
savefig(fig, "fig2c_reward_convergence_heatmap")

# Fig 2d: Reward convergence episode heatmap (for those that converge)
rconv_ep = rconv_df[rconv_df['reward_converged']].groupby(['alpha','sigma'])['reward_conv_episode'].mean().reset_index()
rconv_ep_pivot = rconv_ep.pivot(index='alpha', columns='sigma', values='reward_conv_episode')

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(rconv_ep_pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            cbar_kws={'label': 'Mean Convergence Episode'})
ax.set_title("Mean Episode at Reward Convergence (converged runs only)")
ax.set_xlabel(r"$\sigma$ (preference spread)")
ax.set_ylabel(r"$\alpha$ (friction)")
savefig(fig, "fig2d_reward_convergence_episode")


# =========================================================================
# 3. POLICY SPACE ANALYSIS
# =========================================================================
print("\n=== 3. Policy Space Analysis ===")

# Flatten policy vectors: (4, 27) -> 108-dim
policy_keys = []
policy_flat = []
policy_meta = []

for _, row in df.iterrows():
    key = row['policy_key']
    if key in pv_data:
        pv = pv_data[key]  # (4, 27)
        policy_keys.append(key)
        policy_flat.append(pv.flatten())
        policy_meta.append({
            'alpha': row['alpha'],
            'sigma': row['sigma'],
            'epsilon': row['epsilon'],
            'mean_reward': row['mean_reward'],
        })

policy_matrix = np.array(policy_flat)  # (3750, 108)
meta_df = pd.DataFrame(policy_meta)
print(f"  Policy matrix shape: {policy_matrix.shape}")

# --- PCA ---
print("  Running PCA...")
pca = PCA(n_components=10)
pca_coords = pca.fit_transform(policy_matrix)
print(f"  Explained variance (first 5): {pca.explained_variance_ratio_[:5].round(4)}")
print(f"  Cumulative: {np.cumsum(pca.explained_variance_ratio_[:5]).round(4)}")

# Fig 3a: PCA colored by alpha
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for alpha in sorted(meta_df['alpha'].unique()):
    mask = meta_df['alpha'] == alpha
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               c=ALPHA_COLORS[alpha], label=ALPHA_LABELS[alpha],
               alpha=0.3, s=8, edgecolors='none')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Policy Space (PCA) — Colored by Friction Level")
ax.legend(fontsize=7, markerscale=3)
ax.grid(True, alpha=0.2)

# Fig 3a right: PCA colored by mean_reward
ax = axes[1]
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                     c=meta_df['mean_reward'], cmap='viridis',
                     alpha=0.3, s=8, edgecolors='none')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Policy Space (PCA) — Colored by Mean Reward")
plt.colorbar(scatter, ax=ax, label="Mean Reward")
ax.grid(True, alpha=0.2)

fig.tight_layout()
savefig(fig, "fig3a_pca_policy_space")

# --- t-SNE ---
print("  Running t-SNE (may take a minute)...")
# Subsample for t-SNE performance — take every condition, 10 reps each
subsample_idx = []
for cond in df.drop_duplicates(subset=['alpha','sigma','epsilon']).itertuples():
    mask = (meta_df['alpha'] == cond.alpha) & (meta_df['sigma'] == cond.sigma) & (meta_df['epsilon'] == cond.epsilon)
    indices = np.where(mask)[0]
    subsample_idx.extend(indices[:10])

sub_coords = pca_coords[subsample_idx, :5]  # use first 5 PCs
sub_meta = meta_df.iloc[subsample_idx].reset_index(drop=True)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne_coords = tsne.fit_transform(sub_coords)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for alpha in sorted(sub_meta['alpha'].unique()):
    mask = sub_meta['alpha'] == alpha
    ax.scatter(tsne_coords[mask.values, 0], tsne_coords[mask.values, 1],
               c=ALPHA_COLORS[alpha], label=ALPHA_LABELS[alpha],
               alpha=0.4, s=15, edgecolors='none')
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("Policy Space (t-SNE) — Colored by Friction Level")
ax.legend(fontsize=7, markerscale=2)

ax = axes[1]
scatter = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                     c=sub_meta['mean_reward'], cmap='viridis',
                     alpha=0.4, s=15, edgecolors='none')
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("Policy Space (t-SNE) — Colored by Mean Reward")
plt.colorbar(scatter, ax=ax, label="Mean Reward")

fig.tight_layout()
savefig(fig, "fig3b_tsne_policy_space")

# --- K-means clustering: elbow method ---
print("  Running K-means elbow analysis...")
inertias = []
K_range = range(2, 16)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    km.fit(pca_coords[:, :5])
    inertias.append(km.inertia_)

# Find optimal k using elbow (second derivative)
inertias_arr = np.array(inertias)
d1 = np.diff(inertias_arr)
d2 = np.diff(d1)
optimal_k = list(K_range)[np.argmax(d2) + 2]  # +2 for double diff offset
print(f"  Optimal K (elbow): {optimal_k}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(list(K_range), inertias, 'b-o', markersize=5)
ax1.axvline(optimal_k, color='r', linestyle='--', label=f"Elbow at k={optimal_k}")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("K-Means Elbow Method")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show cluster assignments with optimal k
km_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = km_opt.fit_predict(pca_coords[:, :5])

ax2.scatter(pca_coords[:, 0], pca_coords[:, 1], c=clusters, cmap='tab10',
            alpha=0.3, s=8, edgecolors='none')
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax2.set_title(f"K-Means Clustering (k={optimal_k})")
ax2.grid(True, alpha=0.2)

fig.tight_layout()
savefig(fig, "fig3c_kmeans_elbow")

# --- Cluster composition by alpha ---
cluster_df = pd.DataFrame({'cluster': clusters, 'alpha': meta_df['alpha'], 'mean_reward': meta_df['mean_reward']})
cluster_comp = cluster_df.groupby(['cluster','alpha']).size().unstack(fill_value=0)
cluster_comp = cluster_comp.div(cluster_comp.sum(axis=1), axis=0)  # normalize

fig, ax = plt.subplots(figsize=(8, 5))
cluster_comp.plot(kind='bar', stacked=True, ax=ax,
                  color=[ALPHA_COLORS[a] for a in sorted(meta_df['alpha'].unique())])
ax.set_xlabel("Policy Cluster")
ax.set_ylabel("Proportion")
ax.set_title("Cluster Composition by Friction Level")
ax.legend(title=r"$\alpha$", labels=[f"{a:.1f}" for a in sorted(meta_df['alpha'].unique())])
ax.grid(True, alpha=0.2, axis='y')
savefig(fig, "fig3d_cluster_composition")


# =========================================================================
# 4. AGENT SPECIALIZATION
# =========================================================================
print("\n=== 4. Agent Specialization Analysis ===")

# Per-condition agent reward variance
df['agent_reward_var'] = df[['agent_0_reward','agent_1_reward','agent_2_reward','agent_3_reward']].var(axis=1)
df['agent_reward_range'] = df[['agent_0_reward','agent_1_reward','agent_2_reward','agent_3_reward']].max(axis=1) - \
                           df[['agent_0_reward','agent_1_reward','agent_2_reward','agent_3_reward']].min(axis=1)

# Gini coefficient for agent rewards (treating rewards as allocations after shifting to positive)
def gini(values):
    """Gini coefficient of reward inequality."""
    v = np.array(values, dtype=float)
    v = v - v.min() + 1e-8  # shift to positive
    v = np.sort(v)
    n = len(v)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * v) / (n * np.sum(v))) - (n + 1) / n

df['agent_gini'] = df.apply(
    lambda r: gini([r['agent_0_reward'], r['agent_1_reward'], r['agent_2_reward'], r['agent_3_reward']]),
    axis=1
)

# Fig 4a: Agent variance vs alpha (box plot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

bp_data = [df[df['alpha'] == a]['agent_reward_var'].values for a in sorted(df['alpha'].unique())]
bp = ax1.boxplot(bp_data, positions=range(5), widths=0.6, patch_artist=True,
                 showfliers=False, medianprops=dict(color='black'))
for patch, alpha in zip(bp['boxes'], sorted(df['alpha'].unique())):
    patch.set_facecolor(ALPHA_COLORS[alpha])
    patch.set_alpha(0.7)
ax1.set_xticks(range(5))
ax1.set_xticklabels([f"{a:.1f}" for a in sorted(df['alpha'].unique())])
ax1.set_xlabel(r"$\alpha$ (friction)")
ax1.set_ylabel("Agent Reward Variance")
ax1.set_title("Agent Reward Inequality by Friction Level")
ax1.grid(True, alpha=0.3)

# Fig 4a right: Gini vs alpha
gini_data = [df[df['alpha'] == a]['agent_gini'].values for a in sorted(df['alpha'].unique())]
bp2 = ax2.boxplot(gini_data, positions=range(5), widths=0.6, patch_artist=True,
                  showfliers=False, medianprops=dict(color='black'))
for patch, alpha in zip(bp2['boxes'], sorted(df['alpha'].unique())):
    patch.set_facecolor(ALPHA_COLORS[alpha])
    patch.set_alpha(0.7)
ax2.set_xticks(range(5))
ax2.set_xticklabels([f"{a:.1f}" for a in sorted(df['alpha'].unique())])
ax2.set_xlabel(r"$\alpha$ (friction)")
ax2.set_ylabel("Gini Coefficient")
ax2.set_title("Agent Reward Gini by Friction Level")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
savefig(fig, "fig4a_agent_specialization_alpha")

# Fig 4b: Specialization heatmap alpha x sigma
var_heat = df.groupby(['alpha','sigma'])['agent_reward_var'].median().reset_index()
var_pivot = var_heat.pivot(index='alpha', columns='sigma', values='agent_reward_var')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(var_pivot, annot=True, fmt=".3f", cmap="Reds", ax=ax1,
            cbar_kws={'label': 'Median Agent Variance'})
ax1.set_title(r"Agent Reward Variance ($\alpha \times \sigma$)")
ax1.set_xlabel(r"$\sigma$")
ax1.set_ylabel(r"$\alpha$")

gini_heat = df.groupby(['alpha','sigma'])['agent_gini'].median().reset_index()
gini_pivot = gini_heat.pivot(index='alpha', columns='sigma', values='agent_gini')

sns.heatmap(gini_pivot, annot=True, fmt=".3f", cmap="Reds", ax=ax2,
            cbar_kws={'label': 'Median Gini Coefficient'})
ax2.set_title(r"Agent Gini Coefficient ($\alpha \times \sigma$)")
ax2.set_xlabel(r"$\sigma$")
ax2.set_ylabel(r"$\alpha$")

fig.tight_layout()
savefig(fig, "fig4b_specialization_heatmap")

# Fig 4c: Individual agent rewards for specific conditions
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
conditions = [
    (-0.8, 0.2, 0.5, "Cooperative, Low Noise"),
    (0.0, 0.6, 0.5, "Neutral, Medium Noise"),
    (0.8, 0.2, 0.5, "Adversarial, Low Noise"),
    (-0.8, 1.0, 0.5, "Cooperative, High Noise"),
    (0.0, 1.0, 0.5, "Neutral, High Noise"),
    (0.8, 1.0, 0.5, "Adversarial, High Noise"),
]

agent_colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]

for idx, (alpha, sigma, eps, title) in enumerate(conditions):
    ax = axes[idx // 3, idx % 3]
    sub = df[(df['alpha'] == alpha) & (df['sigma'] == sigma) & (df['epsilon'] == eps)]
    for a in range(4):
        rewards = sub[f'agent_{a}_reward'].values
        ax.hist(rewards, bins=20, alpha=0.4, color=agent_colors[a], label=f"Agent {a}")
    ax.set_title(f"{title}\n" + r"$\alpha=$" + f"{alpha}, $\\sigma=${sigma}", fontsize=9)
    ax.set_xlabel("Reward", fontsize=8)
    if idx == 0:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

fig.suptitle("Per-Agent Reward Distributions Across Conditions", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "fig4c_agent_reward_distributions")

# --- Persistent exploitation detection ---
print("  Detecting persistent exploitation...")
exploit_rows = []
for _, grp in df.groupby(['alpha','sigma','epsilon']):
    agent_means = [grp[f'agent_{a}_reward'].mean() for a in range(4)]
    agent_stds = [grp[f'agent_{a}_reward'].std() for a in range(4)]
    best = np.argmax(agent_means)
    worst = np.argmin(agent_means)
    gap = agent_means[best] - agent_means[worst]

    # t-test between best and worst agents
    t_stat, p_val = stats.ttest_ind(
        grp[f'agent_{best}_reward'].values,
        grp[f'agent_{worst}_reward'].values
    )

    exploit_rows.append({
        'alpha': grp['alpha'].iloc[0],
        'sigma': grp['sigma'].iloc[0],
        'epsilon': grp['epsilon'].iloc[0],
        'best_agent': best,
        'worst_agent': worst,
        'reward_gap': gap,
        't_stat': abs(t_stat),
        'p_value': p_val,
        'significant': p_val < 0.05,
    })

exploit_df = pd.DataFrame(exploit_rows)
sig_frac = exploit_df.groupby('alpha')['significant'].mean()
print(f"  Fraction with significant agent inequality by alpha:")
for a, f in sig_frac.items():
    print(f"    alpha={a:.1f}: {f:.2%}")


# =========================================================================
# 5. REWARD DYNAMICS OVER TRAINING
# =========================================================================
print("\n=== 5. Reward Dynamics Over Training ===")

# Identify interesting conditions
best_cond = df.groupby(['alpha','sigma','epsilon'])['mean_reward'].mean().idxmax()
worst_cond = df.groupby(['alpha','sigma','epsilon'])['mean_reward'].mean().idxmin()

# Phase transition: highest agent variance condition
var_by_cond = df.groupby(['alpha','sigma','epsilon'])['agent_reward_var'].mean()
phase_cond = var_by_cond.idxmax()

print(f"  Best condition:  alpha={best_cond[0]}, sigma={best_cond[1]}, eps={best_cond[2]}")
print(f"  Worst condition: alpha={worst_cond[0]}, sigma={worst_cond[1]}, eps={worst_cond[2]}")
print(f"  Phase transition: alpha={phase_cond[0]}, sigma={phase_cond[1]}, eps={phase_cond[2]}")

# For reward dynamics per agent, we need to reconstruct from per-replication learning curves
# Since LCs are mean over agents, we'll use the per-agent final rewards to infer dynamics
# But we can show individual replication trajectories

def plot_replication_trajectories(alpha, sigma, eps, title, ax):
    """Plot individual replication learning curves for a condition."""
    curves = get_condition_lcs(alpha, sigma, eps)
    if curves is None:
        return

    # Plot individual trajectories in grey
    for r in range(min(30, len(curves))):
        ax.plot(EPISODE_TICKS, curves[r], color='#888888', alpha=0.15, linewidth=0.5)

    # Mean + CI
    mean = curves.mean(axis=0)
    q25 = np.percentile(curves, 25, axis=0)
    q75 = np.percentile(curves, 75, axis=0)

    ax.plot(EPISODE_TICKS, mean, color=ALPHA_COLORS[alpha], linewidth=2, label="Mean")
    ax.fill_between(EPISODE_TICKS, q25, q75, color=ALPHA_COLORS[alpha], alpha=0.2, label="IQR")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Mean Reward", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_replication_trajectories(*best_cond, f"Best: $\\alpha$={best_cond[0]}, $\\sigma$={best_cond[1]}, $\\varepsilon$={best_cond[2]}", axes[0])
plot_replication_trajectories(*worst_cond, f"Worst: $\\alpha$={worst_cond[0]}, $\\sigma$={worst_cond[1]}, $\\varepsilon$={worst_cond[2]}", axes[1])
plot_replication_trajectories(*phase_cond, f"Max Var: $\\alpha$={phase_cond[0]}, $\\sigma$={phase_cond[1]}, $\\varepsilon$={phase_cond[2]}", axes[2])

fig.suptitle("Reward Dynamics: Best, Worst, and Maximum-Variance Conditions", fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, "fig5a_reward_dynamics_key_conditions")

# Fig 5b: Oscillation analysis — coefficient of variation over sliding windows
print("  Analyzing oscillation patterns...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

interesting_conditions = [
    (-0.8, 0.2, 0.5, "Cooperative, Low Noise"),
    (0.0, 0.6, 0.5, "Neutral, Medium Noise"),
    (0.8, 0.2, 0.5, "Adversarial, Low Noise"),
    (-0.8, 1.0, 1.0, "Cooperative, High Noise, High Obs Noise"),
    (0.0, 1.0, 0.0, "Neutral, High Noise, No Obs Noise"),
    (0.8, 1.0, 1.0, "Adversarial, High Noise, High Obs Noise"),
]

for idx, (alpha, sigma, eps, title) in enumerate(interesting_conditions):
    ax = axes[idx // 3, idx % 3]
    curves = get_condition_lcs(alpha, sigma, eps)
    if curves is None:
        continue

    # Sliding window CV (coefficient of variation)
    window = 5
    for r in range(min(5, len(curves))):  # show 5 example reps
        cvs = []
        for w in range(len(curves[r]) - window + 1):
            segment = curves[r, w:w+window]
            cv = segment.std() / (abs(segment.mean()) + 1e-8)
            cvs.append(cv)
        ax.plot(EPISODE_TICKS[window-1:], cvs, alpha=0.5, linewidth=0.8)

    # Mean CV across all reps
    all_cvs = np.zeros((len(curves), len(curves[0]) - window + 1))
    for r in range(len(curves)):
        for w in range(len(curves[r]) - window + 1):
            segment = curves[r, w:w+window]
            all_cvs[r, w] = segment.std() / (abs(segment.mean()) + 1e-8)

    ax.plot(EPISODE_TICKS[window-1:], all_cvs.mean(axis=0), 'k-', linewidth=2, label="Mean CV")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("CV (5-window)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

fig.suptitle("Oscillation Analysis: Coefficient of Variation Over Training", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "fig5b_oscillation_analysis")

# Fig 5c: Improvement rate (derivative of learning curve) by condition
print("  Computing improvement rates...")
fig, ax = plt.subplots(figsize=(10, 6))

for alpha in [-0.8, 0.0, 0.8]:
    all_improvements = []
    for sigma in [0.2, 0.6, 1.0]:
        curves = get_condition_lcs(alpha, sigma, 0.5)
        if curves is not None:
            # Compute first difference (improvement per window)
            diffs = np.diff(curves, axis=1)
            all_improvements.append(diffs.mean(axis=0))

    if all_improvements:
        mean_improvement = np.mean(all_improvements, axis=0)
        ax.plot(EPISODE_TICKS[1:], mean_improvement, color=ALPHA_COLORS[alpha],
                label=ALPHA_LABELS[alpha], linewidth=2)

ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
ax.set_xlabel("Episode")
ax.set_ylabel(r"$\Delta$ Reward per Window")
ax.set_title("Learning Rate (Reward Improvement) Over Training")
ax.legend()
ax.grid(True, alpha=0.3)
savefig(fig, "fig5c_improvement_rate")


# =========================================================================
# SUMMARY STATISTICS
# =========================================================================
print("\n=== Summary Statistics ===")

# Collect all stats for the report
summary = {}

# Overall
summary['total_conditions'] = 125
summary['total_replications'] = 3750
summary['mean_reward'] = df['mean_reward'].mean()
summary['std_reward'] = df['mean_reward'].std()

# By alpha
alpha_stats = df.groupby('alpha')['mean_reward'].agg(['mean','std','min','max'])
summary['alpha_stats'] = alpha_stats

# Best/worst conditions
summary['best_cond'] = best_cond
summary['worst_cond'] = worst_cond
summary['best_reward'] = df.groupby(['alpha','sigma','epsilon'])['mean_reward'].mean().max()
summary['worst_reward'] = df.groupby(['alpha','sigma','epsilon'])['mean_reward'].mean().min()

# Convergence
summary['policy_conv_rate'] = df['converged'].mean()
summary['reward_conv_rate'] = rconv_df['reward_converged'].mean()

# Clustering
summary['n_clusters'] = optimal_k
summary['pca_var_explained'] = np.cumsum(pca.explained_variance_ratio_[:5])

# Agent specialization
summary['mean_agent_var_alpha0'] = df[df['alpha']==0.0]['agent_reward_var'].mean()
summary['mean_agent_var_alpha08'] = df[df['alpha']==0.8]['agent_reward_var'].mean()
summary['mean_agent_var_alpha_n08'] = df[df['alpha']==-0.8]['agent_reward_var'].mean()

# Write summary to file for report generation
import json
summary_serializable = {
    'total_conditions': 125,
    'total_replications': 3750,
    'mean_reward': float(summary['mean_reward']),
    'std_reward': float(summary['std_reward']),
    'best_condition': {'alpha': float(best_cond[0]), 'sigma': float(best_cond[1]), 'epsilon': float(best_cond[2])},
    'worst_condition': {'alpha': float(worst_cond[0]), 'sigma': float(worst_cond[1]), 'epsilon': float(worst_cond[2])},
    'best_reward': float(summary['best_reward']),
    'worst_reward': float(summary['worst_reward']),
    'policy_convergence_rate': float(summary['policy_conv_rate']),
    'reward_convergence_rate': float(summary['reward_conv_rate']),
    'n_policy_clusters': int(optimal_k),
    'pca_variance_5pc': list(summary['pca_var_explained'].round(4)),
    'alpha_stats': alpha_stats.to_dict(),
    'agent_var_by_alpha': {
        '-0.8': float(summary['mean_agent_var_alpha_n08']),
        '0.0': float(summary['mean_agent_var_alpha0']),
        '0.8': float(summary['mean_agent_var_alpha08']),
    },
    'exploit_significance': exploit_df.groupby('alpha')['significant'].mean().to_dict(),
    'speed_by_alpha': speed_df.groupby('alpha')['mean_episodes_to_80'].mean().to_dict(),
    'stability_by_alpha': stability_df.groupby('alpha')['late_std'].mean().to_dict(),
}

with open(OUT / 'summary_stats.json', 'w') as f:
    json.dump(summary_serializable, f, indent=2, default=str)

print(f"\n  Summary saved to {OUT / 'summary_stats.json'}")
print("\nDone! All figures saved to:", OUT)
