#!/usr/bin/env python3
"""
MARL 5x5x5 Factorial Experiment — Full Analysis
=================================================
Analyzes friction (alpha) x noise (sigma) x exploration (epsilon) effects
on multi-agent coordination outcomes.

For: Axiom of Consent paper appendix
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / 'replication_results.csv'
OUT_DIR = Path(__file__).parent
FIGSIZE_SINGLE = (6, 4.5)
FIGSIZE_WIDE = (10, 4.5)
FIGSIZE_HEATMAP = (7, 5.5)
FIGSIZE_PANEL = (16, 10)
DPI = 300

# Burgundy color scheme matching paper template
BURGUNDY = '#800020'
BURGUNDY_LIGHT = '#A0304A'
BURGUNDY_DARK = '#5A0015'
PALETTE = sns.color_palette([BURGUNDY_DARK, BURGUNDY, BURGUNDY_LIGHT, '#C06070', '#D89098'])

sns.set_context('paper', font_scale=1.2)
sns.set_style('whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUT_DIR / f'{name}.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT_DIR / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}.png / .pdf')


def gini_coefficient(values):
    """Compute Gini coefficient for an array of values."""
    values = np.array(values, dtype=float)
    # Shift to non-negative for Gini calculation
    values = values - values.min() + 1e-10
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1.0) / n


# ══════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print('='*70)
print('MARL 5x5x5 FACTORIAL — FULL ANALYSIS')
print('='*70)

df = pd.read_csv(DATA_PATH)
print(f'\nLoaded {len(df)} rows, {df.columns.tolist()}')
print(f'Alpha levels:   {sorted(df.alpha.unique())}')
print(f'Sigma levels:   {sorted(df.sigma.unique())}')
print(f'Epsilon levels: {sorted(df.epsilon.unique())}')
print(f'Replications per condition: {df.groupby(["alpha","sigma","epsilon"]).size().unique()}')

agent_cols = [c for c in df.columns if c.startswith('agent_') and c.endswith('_reward')]
print(f'Agent columns: {agent_cols}')

# ══════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('1. DESCRIPTIVE STATISTICS')
print('='*70)

# Per-condition summary
cond_stats = df.groupby(['alpha', 'sigma', 'epsilon']).agg(
    reward_mean=('mean_reward', 'mean'),
    reward_std=('mean_reward', 'std'),
    reward_ci95=('mean_reward', lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    conv_mean=('convergence_time', 'mean'),
    conv_std=('convergence_time', 'std'),
    conv_ci95=('convergence_time', lambda x: 1.96 * x.std() / np.sqrt(len(x))),
    n=('mean_reward', 'count')
).reset_index()

# Best and worst
best = cond_stats.loc[cond_stats.reward_mean.idxmax()]
worst = cond_stats.loc[cond_stats.reward_mean.idxmin()]
print(f'\nBest condition:  alpha={best.alpha}, sigma={best.sigma}, eps={best.epsilon}')
print(f'  Mean reward: {best.reward_mean:.4f} +/- {best.reward_ci95:.4f}')
print(f'  Conv time:   {best.conv_mean:.1f} +/- {best.conv_ci95:.1f}')
print(f'\nWorst condition: alpha={worst.alpha}, sigma={worst.sigma}, eps={worst.epsilon}')
print(f'  Mean reward: {worst.reward_mean:.4f} +/- {worst.reward_ci95:.4f}')
print(f'  Conv time:   {worst.conv_mean:.1f} +/- {worst.conv_ci95:.1f}')

# Global stats
print(f'\nGlobal mean_reward: {df.mean_reward.mean():.4f} (std={df.mean_reward.std():.4f})')
print(f'Global conv_time:   {df.convergence_time.mean():.1f} (std={df.convergence_time.std():.1f})')

# ── Figure 1: Reward distribution histogram ──
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

axes[0].hist(df.mean_reward, bins=60, color=BURGUNDY, alpha=0.8, edgecolor='white', linewidth=0.3)
axes[0].axvline(df.mean_reward.mean(), color='black', linestyle='--', linewidth=1, label=f'Mean = {df.mean_reward.mean():.3f}')
axes[0].axvline(df.mean_reward.median(), color='gray', linestyle=':', linewidth=1, label=f'Median = {df.mean_reward.median():.3f}')
axes[0].set_xlabel('Mean Reward')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Mean Reward (all conditions)')
axes[0].legend(frameon=True, facecolor='white')

axes[1].hist(df.convergence_time, bins=60, color=BURGUNDY_DARK, alpha=0.8, edgecolor='white', linewidth=0.3)
axes[1].axvline(df.convergence_time.mean(), color='black', linestyle='--', linewidth=1, label=f'Mean = {df.convergence_time.mean():.0f}')
axes[1].set_xlabel('Convergence Time (steps)')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Convergence Time (all conditions)')
axes[1].legend(frameon=True, facecolor='white')

fig.tight_layout()
save_fig(fig, 'fig01_distributions')

# Save condition stats
cond_stats.to_csv(OUT_DIR / 'condition_statistics.csv', index=False, float_format='%.6f')
print('  Saved: condition_statistics.csv')


# ══════════════════════════════════════════════════════════════════════════
# 2. MAIN EFFECTS
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('2. MAIN EFFECTS')
print('='*70)

params = {'alpha': r'$\alpha$ (Friction)', 'sigma': r'$\sigma$ (Noise)', 'epsilon': r'$\varepsilon$ (Exploration)'}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for i, (param, label) in enumerate(params.items()):
    for j, (outcome, outcome_label) in enumerate([('mean_reward', 'Mean Reward'), ('convergence_time', 'Convergence Time')]):
        ax = axes[j, i]
        grouped = df.groupby(param)[outcome]
        means = grouped.mean()
        sems = grouped.sem()

        ax.errorbar(means.index, means.values, yerr=1.96*sems.values,
                     fmt='o-', color=BURGUNDY, capsize=4, capthick=1.5,
                     markersize=7, linewidth=2, markeredgecolor='white', markeredgewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel(outcome_label)
        ax.grid(True, alpha=0.3)

        # Compute effect range
        effect_range = means.max() - means.min()
        ax.set_title(f'{label}\n(range = {effect_range:.4f})')

fig.suptitle('Main Effects on Reward and Convergence Time', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'fig02_main_effects')

# Quantify main effects
print('\nMain effect magnitudes (range of marginal means):')
for param, label in params.items():
    for outcome in ['mean_reward', 'convergence_time']:
        means = df.groupby(param)[outcome].mean()
        effect = means.max() - means.min()
        print(f'  {param:>8} -> {outcome:>18}: range = {effect:.4f}')


# ══════════════════════════════════════════════════════════════════════════
# 3. INTERACTION HEATMAPS (KEY OUTPUT)
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('3. PAIRWISE INTERACTION HEATMAPS')
print('='*70)

interactions = [
    ('alpha', 'sigma', r'$\alpha$ (Friction)', r'$\sigma$ (Noise)'),
    ('alpha', 'epsilon', r'$\alpha$ (Friction)', r'$\varepsilon$ (Exploration)'),
    ('sigma', 'epsilon', r'$\sigma$ (Noise)', r'$\varepsilon$ (Exploration)'),
]

for outcome, outcome_label, cmap_name, fignum in [
    ('mean_reward', 'Mean Reward', 'RdYlGn', '03a'),
    ('convergence_time', 'Convergence Time', 'RdYlGn_r', '03b')
]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for k, (p1, p2, l1, l2) in enumerate(interactions):
        pivot = df.groupby([p1, p2])[outcome].mean().unstack()

        # Ensure proper ordering
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot[sorted(pivot.columns)]

        im = sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap_name,
                         ax=axes[k], linewidths=0.5, linecolor='white',
                         cbar_kws={'label': outcome_label, 'shrink': 0.8},
                         annot_kws={'size': 9})
        axes[k].set_xlabel(l2)
        axes[k].set_ylabel(l1)
        axes[k].set_title(f'{l1} x {l2}')
        axes[k].invert_yaxis()

    fig.suptitle(f'Pairwise Interactions — {outcome_label}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, f'fig{fignum}_heatmaps_{outcome}')


# ══════════════════════════════════════════════════════════════════════════
# 4. THREE-WAY INTERACTION
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('4. THREE-WAY INTERACTION (alpha x sigma | epsilon)')
print('='*70)

epsilon_levels = sorted(df.epsilon.unique())

fig, axes = plt.subplots(1, 5, figsize=(24, 5))

# Compute global color limits for consistency
all_pivots = []
for eps in epsilon_levels:
    sub = df[df.epsilon == eps]
    pivot = sub.groupby(['alpha', 'sigma']).mean_reward.mean().unstack()
    all_pivots.append(pivot)

vmin = min(p.values.min() for p in all_pivots)
vmax = max(p.values.max() for p in all_pivots)

for i, (eps, pivot) in enumerate(zip(epsilon_levels, all_pivots)):
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=axes[i], linewidths=0.5, linecolor='white',
                vmin=vmin, vmax=vmax,
                cbar=(i == 4),
                cbar_kws={'label': 'Mean Reward', 'shrink': 0.8} if i == 4 else {},
                annot_kws={'size': 8})
    axes[i].set_title(f'$\\varepsilon$ = {eps}')
    axes[i].set_xlabel(r'$\sigma$ (Noise)')
    axes[i].set_ylabel(r'$\alpha$ (Friction)' if i == 0 else '')
    axes[i].invert_yaxis()

fig.suptitle(r'Three-Way Interaction: $\alpha \times \sigma$ conditioned on $\varepsilon$',
             fontsize=14, fontweight='bold', y=1.04)
fig.tight_layout()
save_fig(fig, 'fig04_threeway_alpha_sigma_by_epsilon')

# Identify sweet spot
print('\nCondition-level reward ranking (top 10):')
top10 = cond_stats.nlargest(10, 'reward_mean')
for _, row in top10.iterrows():
    print(f'  alpha={row.alpha:+.1f}, sigma={row.sigma:.1f}, eps={row.epsilon:.2f}: '
          f'reward={row.reward_mean:.4f} +/- {row.reward_ci95:.4f}')


# ══════════════════════════════════════════════════════════════════════════
# 5. AGENT INEQUALITY (GINI)
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('5. AGENT INEQUALITY (GINI COEFFICIENT)')
print('='*70)

# Compute Gini for each row
df['gini'] = df.apply(lambda r: gini_coefficient([r[c] for c in agent_cols]), axis=1)

# Gini summary by condition
gini_stats = df.groupby(['alpha', 'sigma', 'epsilon']).agg(
    gini_mean=('gini', 'mean'),
    gini_std=('gini', 'std'),
    gini_ci95=('gini', lambda x: 1.96 * x.std() / np.sqrt(len(x)))
).reset_index()

# ── Figure 5a: Gini vs Alpha ──
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
gini_by_alpha = df.groupby('alpha')['gini']
means = gini_by_alpha.mean()
sems = gini_by_alpha.sem()
ax.errorbar(means.index, means.values, yerr=1.96*sems.values,
            fmt='o-', color=BURGUNDY, capsize=5, capthick=1.5,
            markersize=8, linewidth=2, markeredgecolor='white')
ax.set_xlabel(r'$\alpha$ (Friction)')
ax.set_ylabel('Gini Coefficient')
ax.set_title('Agent Reward Inequality vs. Friction')
ax.grid(True, alpha=0.3)
save_fig(fig, 'fig05a_gini_vs_alpha')

# ── Figure 5b: Gini heatmap alpha x sigma ──
fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
gini_pivot = df.groupby(['alpha', 'sigma'])['gini'].mean().unstack()
gini_pivot = gini_pivot.sort_index(ascending=True)
gini_pivot = gini_pivot[sorted(gini_pivot.columns)]

sns.heatmap(gini_pivot, annot=True, fmt='.4f', cmap='YlOrRd',
            ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Gini Coefficient', 'shrink': 0.8},
            annot_kws={'size': 10})
ax.set_xlabel(r'$\sigma$ (Noise)')
ax.set_ylabel(r'$\alpha$ (Friction)')
ax.set_title(r'Agent Inequality: $\alpha \times \sigma$')
ax.invert_yaxis()
save_fig(fig, 'fig05b_gini_heatmap_alpha_sigma')

# ── Figure 5c: Gini vs alpha colored by sigma ──
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
sigma_levels = sorted(df.sigma.unique())
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_levels)))
for sig, color in zip(sigma_levels, colors):
    sub = df[df.sigma == sig].groupby('alpha')['gini']
    means = sub.mean()
    sems = sub.sem()
    ax.errorbar(means.index, means.values, yerr=1.96*sems.values,
                fmt='o-', color=color, capsize=3, markersize=5, linewidth=1.5,
                label=f'$\\sigma$={sig}')
ax.set_xlabel(r'$\alpha$ (Friction)')
ax.set_ylabel('Gini Coefficient')
ax.set_title('Agent Inequality: Friction x Noise Interaction')
ax.legend(frameon=True, facecolor='white')
ax.grid(True, alpha=0.3)
save_fig(fig, 'fig05c_gini_alpha_by_sigma')

print(f'\nGini range: [{df.gini.min():.4f}, {df.gini.max():.4f}]')
print(f'Mean Gini:  {df.gini.mean():.4f} (std={df.gini.std():.4f})')
gini_by_a = df.groupby('alpha')['gini'].mean()
print(f'Gini by alpha:\n{gini_by_a.to_string()}')


# ══════════════════════════════════════════════════════════════════════════
# 6. ANOVA / REGRESSION
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('6. ANOVA / REGRESSION')
print('='*70)

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Three-way ANOVA
    print('\n--- Three-Way ANOVA: mean_reward ~ alpha * sigma * epsilon ---')
    model = ols('mean_reward ~ C(alpha) * C(sigma) * C(epsilon)', data=df).fit()
    anova_table = anova_lm(model, typ=2)

    # Compute eta-squared
    ss_total = anova_table['sum_sq'].sum()
    anova_table['eta_sq'] = anova_table['sum_sq'] / ss_total
    anova_table['eta_sq_pct'] = anova_table['eta_sq'] * 100

    print(anova_table.to_string(float_format=lambda x: f'{x:.4f}'))

    # Save ANOVA table
    anova_table.to_csv(OUT_DIR / 'anova_table.csv', float_format='%.6f')
    print('\n  Saved: anova_table.csv')

    # OLS regression with main effects + 2-way interactions
    print('\n--- OLS Regression: main effects + 2-way interactions ---')
    reg_model = ols('mean_reward ~ alpha + sigma + epsilon + '
                    'alpha:sigma + alpha:epsilon + sigma:epsilon', data=df).fit()
    print(reg_model.summary().tables[0])
    print(reg_model.summary().tables[1])

    # Save regression summary
    with open(OUT_DIR / 'regression_summary.txt', 'w') as f:
        f.write(str(reg_model.summary()))
    print('\n  Saved: regression_summary.txt')

    # ANOVA for convergence time too
    print('\n--- Three-Way ANOVA: convergence_time ~ alpha * sigma * epsilon ---')
    model_conv = ols('convergence_time ~ C(alpha) * C(sigma) * C(epsilon)', data=df).fit()
    anova_conv = anova_lm(model_conv, typ=2)
    ss_total_conv = anova_conv['sum_sq'].sum()
    anova_conv['eta_sq'] = anova_conv['sum_sq'] / ss_total_conv
    anova_conv['eta_sq_pct'] = anova_conv['eta_sq'] * 100
    print(anova_conv.to_string(float_format=lambda x: f'{x:.4f}'))
    anova_conv.to_csv(OUT_DIR / 'anova_convergence.csv', float_format='%.6f')

    HAS_STATSMODELS = True
except ImportError:
    print('WARNING: statsmodels not installed. Skipping ANOVA/regression.')
    print('Install with: pip install statsmodels')
    HAS_STATSMODELS = False


# ══════════════════════════════════════════════════════════════════════════
# 7. SUPPLEMENTARY: CONVERGENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('7. SUPPLEMENTARY: CONVERGENCE PATTERNS')
print('='*70)

# What fraction converged before max steps?
max_steps = df.convergence_time.max()
df['converged'] = df.convergence_time < max_steps
conv_rate = df.groupby(['alpha', 'sigma', 'epsilon'])['converged'].mean().reset_index()
conv_rate.columns = ['alpha', 'sigma', 'epsilon', 'convergence_rate']

print(f'\nMax convergence time: {max_steps}')
print(f'Overall convergence rate: {df.converged.mean():.3f}')

# Convergence rate heatmap alpha x sigma
fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
conv_pivot = df.groupby(['alpha', 'sigma'])['converged'].mean().unstack()
conv_pivot = conv_pivot.sort_index(ascending=True)
conv_pivot = conv_pivot[sorted(conv_pivot.columns)]

sns.heatmap(conv_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Convergence Rate', 'shrink': 0.8},
            annot_kws={'size': 10})
ax.set_xlabel(r'$\sigma$ (Noise)')
ax.set_ylabel(r'$\alpha$ (Friction)')
ax.set_title(r'Convergence Rate: $\alpha \times \sigma$')
ax.invert_yaxis()
save_fig(fig, 'fig06_convergence_rate_heatmap')


# ══════════════════════════════════════════════════════════════════════════
# 8. COMPOSITE FIGURE FOR PAPER
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('8. COMPOSITE FIGURE (paper-ready)')
print('='*70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: Main effects
for i, (param, label) in enumerate(params.items()):
    ax = fig.add_subplot(gs[0, i])
    grouped = df.groupby(param)['mean_reward']
    means = grouped.mean()
    sems = grouped.sem()
    ax.errorbar(means.index, means.values, yerr=1.96*sems.values,
                fmt='o-', color=BURGUNDY, capsize=4, capthick=1.5,
                markersize=7, linewidth=2, markeredgecolor='white', markeredgewidth=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel('Mean Reward' if i == 0 else '')
    ax.set_title(f'({chr(97+i)}) Main Effect: {label}')
    ax.grid(True, alpha=0.3)

# Row 2: Key heatmaps
# alpha x sigma (mean_reward)
ax = fig.add_subplot(gs[1, 0])
pivot = df.groupby(['alpha', 'sigma'])['mean_reward'].mean().unstack()
pivot = pivot.sort_index(ascending=True)[sorted(pivot.columns)]
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$\alpha$')
ax.set_title(r'(d) $\alpha \times \sigma$: Reward')
ax.invert_yaxis()

# alpha x sigma (Gini)
ax = fig.add_subplot(gs[1, 1])
sns.heatmap(gini_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$\alpha$')
ax.set_title(r'(e) $\alpha \times \sigma$: Inequality (Gini)')
ax.invert_yaxis()

# alpha x sigma (convergence rate)
ax = fig.add_subplot(gs[1, 2])
sns.heatmap(conv_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$\alpha$')
ax.set_title(r'(f) $\alpha \times \sigma$: Convergence Rate')
ax.invert_yaxis()

# Row 3: Three-way (3 of 5 epsilon levels)
epsilon_subset = [0.0, 0.5, 1.0]
for i, eps in enumerate(epsilon_subset):
    ax = fig.add_subplot(gs[2, i])
    sub = df[df.epsilon == eps]
    pivot = sub.groupby(['alpha', 'sigma']).mean_reward.mean().unstack()
    pivot = pivot.sort_index(ascending=True)[sorted(pivot.columns)]
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                linewidths=0.5, linecolor='white', vmin=vmin, vmax=vmax,
                cbar=(i == 2), cbar_kws={'shrink': 0.8} if i == 2 else {},
                annot_kws={'size': 8})
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\alpha$' if i == 0 else '')
    ax.set_title(f'(g) $\\varepsilon$ = {eps}' if i == 0 else f'({chr(104+i-1)}) $\\varepsilon$ = {eps}')
    ax.invert_yaxis()

fig.suptitle('MARL 5x5x5 Factorial: Friction, Noise, and Exploration Effects',
             fontsize=15, fontweight='bold', y=1.01)
save_fig(fig, 'fig_composite_paper')


# ══════════════════════════════════════════════════════════════════════════
# 9. GENERATE REPORT
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('9. GENERATING ANALYSIS REPORT')
print('='*70)

# Collect key stats for report
alpha_effect_reward = df.groupby('alpha')['mean_reward'].mean()
sigma_effect_reward = df.groupby('sigma')['mean_reward'].mean()
epsilon_effect_reward = df.groupby('epsilon')['mean_reward'].mean()

report = f"""# MARL 5x5x5 Factorial Experiment — Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Data:** 3,750 observations (125 conditions x 30 replications)
**Design:** alpha (friction) x sigma (noise) x epsilon (exploration), full factorial

---

## 1. Parameter Space

| Parameter | Symbol | Levels | Interpretation |
|-----------|--------|--------|----------------|
| Friction  | alpha  | [-0.8, -0.4, 0.0, 0.4, 0.8] | Negative = cooperative, Positive = adversarial |
| Noise     | sigma  | [0.2, 0.4, 0.6, 0.8, 1.0] | Environment stochasticity |
| Exploration | epsilon | [0.0, 0.25, 0.5, 0.75, 1.0] | Random action probability |

## 2. Descriptive Statistics

### Global Summary
- **Mean reward:** {df.mean_reward.mean():.4f} (SD = {df.mean_reward.std():.4f})
- **Mean convergence time:** {df.convergence_time.mean():.1f} (SD = {df.convergence_time.std():.1f})
- **Overall convergence rate:** {df.converged.mean():.1%}

### Best Condition
- alpha = {best.alpha}, sigma = {best.sigma}, epsilon = {best.epsilon}
- Mean reward = {best.reward_mean:.4f} (95% CI: +/- {best.reward_ci95:.4f})
- Convergence time = {best.conv_mean:.1f}

### Worst Condition
- alpha = {worst.alpha}, sigma = {worst.sigma}, epsilon = {worst.epsilon}
- Mean reward = {worst.reward_mean:.4f} (95% CI: +/- {worst.reward_ci95:.4f})
- Convergence time = {worst.conv_mean:.1f}

**Figure:** `fig01_distributions.pdf`

## 3. Main Effects

### Mean Reward (marginal means)
| Level | alpha (friction) | sigma (noise) | epsilon (exploration) |
|-------|-----------------|---------------|----------------------|
"""

# Build main effects table
for level_idx in range(5):
    alpha_vals = sorted(df.alpha.unique())
    sigma_vals = sorted(df.sigma.unique())
    eps_vals = sorted(df.epsilon.unique())
    a_val = alpha_effect_reward[alpha_vals[level_idx]]
    s_val = sigma_effect_reward[sigma_vals[level_idx]]
    e_val = epsilon_effect_reward[eps_vals[level_idx]]
    report += f'| {level_idx} | {alpha_vals[level_idx]:+.1f}: {a_val:.4f} | {sigma_vals[level_idx]:.1f}: {s_val:.4f} | {eps_vals[level_idx]:.2f}: {e_val:.4f} |\n'

# Effect magnitudes
a_range = alpha_effect_reward.max() - alpha_effect_reward.min()
s_range = sigma_effect_reward.max() - sigma_effect_reward.min()
e_range = epsilon_effect_reward.max() - epsilon_effect_reward.min()

strongest = 'alpha (friction)' if a_range >= max(s_range, e_range) else ('sigma (noise)' if s_range >= e_range else 'epsilon (exploration)')

report += f"""
### Effect Magnitudes (range of marginal means)
- **alpha:** {a_range:.4f}
- **sigma:** {s_range:.4f}
- **epsilon:** {e_range:.4f}
- **Strongest main effect:** {strongest}

**Figure:** `fig02_main_effects.pdf`

## 4. Pairwise Interactions

Six heatmaps generated showing mean values of reward and convergence time for every pair of parameters, marginalizing over the third.

**Figures:** `fig03a_heatmaps_mean_reward.pdf`, `fig03b_heatmaps_convergence_time.pdf`

## 5. Three-Way Interaction

The alpha x sigma heatmap is conditioned on each epsilon level (5 panels), revealing how the friction-noise interaction shifts with exploration.

"""

# Find sweet spot region
sweet = cond_stats.nlargest(5, 'reward_mean')
report += "### Top 5 Conditions (Reward)\n"
report += "| alpha | sigma | epsilon | Mean Reward | 95% CI |\n"
report += "|-------|-------|---------|-------------|--------|\n"
for _, row in sweet.iterrows():
    report += f'| {row.alpha:+.1f} | {row.sigma:.1f} | {row.epsilon:.2f} | {row.reward_mean:.4f} | +/- {row.reward_ci95:.4f} |\n'

report += f"""
**Figure:** `fig04_threeway_alpha_sigma_by_epsilon.pdf`

## 6. Agent Inequality (Gini Coefficient)

The Gini coefficient measures reward inequality among the 4 agents within each condition.

- **Mean Gini:** {df.gini.mean():.4f} (SD = {df.gini.std():.4f})
- **Range:** [{df.gini.min():.4f}, {df.gini.max():.4f}]

### Gini by Alpha (Friction)
"""
for a in sorted(df.alpha.unique()):
    g = df[df.alpha == a]['gini'].mean()
    report += f'- alpha = {a:+.1f}: Gini = {g:.4f}\n'

report += """
**Figures:** `fig05a_gini_vs_alpha.pdf`, `fig05b_gini_heatmap_alpha_sigma.pdf`, `fig05c_gini_alpha_by_sigma.pdf`

## 7. Convergence Analysis

"""
report += f"- **Overall convergence rate:** {df.converged.mean():.1%}\n"
report += f"- **Max convergence time observed:** {max_steps}\n\n"
report += "**Figure:** `fig06_convergence_rate_heatmap.pdf`\n"

report += """
## 8. ANOVA / Regression

"""

if HAS_STATSMODELS:
    report += "### Three-Way ANOVA: mean_reward ~ alpha * sigma * epsilon (Type II)\n\n"
    report += "| Source | SS | df | F | p | eta-squared |\n"
    report += "|--------|----|----|---|---|-------------|\n"
    for idx, row in anova_table.iterrows():
        source = str(idx).replace('C(alpha)', 'alpha').replace('C(sigma)', 'sigma').replace('C(epsilon)', 'epsilon')
        report += f'| {source} | {row["sum_sq"]:.2f} | {row["df"]:.0f} | {row["F"]:.2f} | {row["PR(>F)"]:.2e} | {row["eta_sq"]:.4f} ({row["eta_sq_pct"]:.1f}%) |\n'

    report += f"\n### OLS Regression (R-squared = {reg_model.rsquared:.4f}, Adj. R-squared = {reg_model.rsquared_adj:.4f})\n\n"
    report += "| Predictor | Coefficient | SE | t | p |\n"
    report += "|-----------|------------|-----|---|---|\n"
    for name, coef, se, t, p in zip(reg_model.params.index, reg_model.params, reg_model.bse, reg_model.tvalues, reg_model.pvalues):
        report += f'| {name} | {coef:.4f} | {se:.4f} | {t:.2f} | {p:.2e} |\n'

    report += "\n**Files:** `anova_table.csv`, `anova_convergence.csv`, `regression_summary.txt`\n"
else:
    report += "*statsmodels not installed — ANOVA/regression skipped.*\n"

report += f"""
## 9. Composite Figure

A single publication-ready figure combining main effects (row 1), key heatmaps (row 2), and three-way interaction panels (row 3).

**Figure:** `fig_composite_paper.pdf`

## 10. Files Generated

### Figures
| File | Description |
|------|-------------|
| fig01_distributions.pdf | Reward and convergence time distributions |
| fig02_main_effects.pdf | Main effects with 95% CI |
| fig03a_heatmaps_mean_reward.pdf | Pairwise interaction heatmaps (reward) |
| fig03b_heatmaps_convergence_time.pdf | Pairwise interaction heatmaps (convergence) |
| fig04_threeway_alpha_sigma_by_epsilon.pdf | Three-way interaction panels |
| fig05a_gini_vs_alpha.pdf | Inequality vs friction |
| fig05b_gini_heatmap_alpha_sigma.pdf | Inequality heatmap |
| fig05c_gini_alpha_by_sigma.pdf | Inequality by friction, colored by noise |
| fig06_convergence_rate_heatmap.pdf | Convergence rate heatmap |
| fig_composite_paper.pdf | Publication-ready composite |

### Data
| File | Description |
|------|-------------|
| condition_statistics.csv | Per-condition summary statistics |
| anova_table.csv | Three-way ANOVA results (reward) |
| anova_convergence.csv | Three-way ANOVA results (convergence) |
| regression_summary.txt | Full OLS regression output |
"""

with open(OUT_DIR / 'ANALYSIS_REPORT.md', 'w') as f:
    f.write(report)
print('  Saved: ANALYSIS_REPORT.md')


print('\n' + '='*70)
print('ANALYSIS COMPLETE')
print('='*70)
print(f'All outputs in: {OUT_DIR}')
