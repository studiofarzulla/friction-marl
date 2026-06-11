# MARL 5x5x5 Factorial Experiment — Analysis Report

**Generated:** 2026-02-13 18:07
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
- **Mean reward:** -1.7242 (SD = 1.2517)
- **Mean convergence time:** 1198.7 (SD = 14.5)
- **Overall convergence rate:** 0.9%

### Best Condition
- alpha = 0.4, sigma = 0.2, epsilon = 0.0
- Mean reward = -0.4253 (95% CI: +/- 0.0515)
- Convergence time = 1200.0

### Worst Condition
- alpha = 0.0, sigma = 1.0, epsilon = 0.75
- Mean reward = -4.5112 (95% CI: +/- 0.5890)
- Convergence time = 1200.0

**Figure:** `fig01_distributions.pdf`

## 3. Main Effects

### Mean Reward (marginal means)
| Level | alpha (friction) | sigma (noise) | epsilon (exploration) |
|-------|-----------------|---------------|----------------------|
| 0 | -0.8: -1.6310 | 0.2: -0.5964 | 0.00: -1.5568 |
| 1 | -0.4: -1.4531 | 0.4: -1.1678 | 0.25: -1.7148 |
| 2 | +0.0: -2.4857 | 0.6: -1.7457 | 0.50: -1.6597 |
| 3 | +0.4: -1.4564 | 0.8: -2.2897 | 0.75: -1.8356 |
| 4 | +0.8: -1.5947 | 1.0: -2.8212 | 1.00: -1.8540 |

### Effect Magnitudes (range of marginal means)
- **alpha:** 1.0327
- **sigma:** 2.2248
- **epsilon:** 0.2971
- **Strongest main effect:** sigma (noise)

**Figure:** `fig02_main_effects.pdf`

## 4. Pairwise Interactions

Six heatmaps generated showing mean values of reward and convergence time for every pair of parameters, marginalizing over the third.

**Figures:** `fig03a_heatmaps_mean_reward.pdf`, `fig03b_heatmaps_convergence_time.pdf`

## 5. Three-Way Interaction

The alpha x sigma heatmap is conditioned on each epsilon level (5 panels), revealing how the friction-noise interaction shifts with exploration.

### Top 5 Conditions (Reward)
| alpha | sigma | epsilon | Mean Reward | 95% CI |
|-------|-------|---------|-------------|--------|
| +0.4 | 0.2 | 0.00 | -0.4253 | +/- 0.0515 |
| -0.4 | 0.2 | 0.50 | -0.4471 | +/- 0.0522 |
| -0.8 | 0.2 | 0.00 | -0.4475 | +/- 0.1024 |
| -0.4 | 0.2 | 0.25 | -0.4528 | +/- 0.0629 |
| -0.4 | 0.2 | 0.75 | -0.4753 | +/- 0.0505 |

**Figure:** `fig04_threeway_alpha_sigma_by_epsilon.pdf`

## 6. Agent Inequality (Gini Coefficient)

The Gini coefficient measures reward inequality among the 4 agents within each condition.

- **Mean Gini:** 0.3966 (SD = 0.0905)
- **Range:** [0.2518, 0.7469]

### Gini by Alpha (Friction)
- alpha = -0.8: Gini = 0.4122
- alpha = -0.4: Gini = 0.3883
- alpha = +0.0: Gini = 0.3852
- alpha = +0.4: Gini = 0.3866
- alpha = +0.8: Gini = 0.4108

**Figures:** `fig05a_gini_vs_alpha.pdf`, `fig05b_gini_heatmap_alpha_sigma.pdf`, `fig05c_gini_alpha_by_sigma.pdf`

## 7. Convergence Analysis

- **Overall convergence rate:** 0.9%
- **Max convergence time observed:** 1200.0

**Figure:** `fig06_convergence_rate_heatmap.pdf`

## 8. ANOVA / Regression

### Three-Way ANOVA: mean_reward ~ alpha * sigma * epsilon (Type II)

| Source | SS | df | F | p | eta-squared |
|--------|----|----|---|---|-------------|
| alpha | 562.98 | 4 | 186.76 | 1.00e-145 | 0.0958 (9.6%) |
| sigma | 2328.87 | 4 | 772.55 | 0.00e+00 | 0.3965 (39.6%) |
| epsilon | 46.13 | 4 | 15.30 | 2.05e-12 | 0.0079 (0.8%) |
| alpha:sigma | 123.48 | 16 | 10.24 | 6.35e-26 | 0.0210 (2.1%) |
| alpha:epsilon | 13.24 | 16 | 1.10 | 3.50e-01 | 0.0023 (0.2%) |
| sigma:epsilon | 22.25 | 16 | 1.85 | 2.10e-02 | 0.0038 (0.4%) |
| alpha:sigma:epsilon | 45.11 | 64 | 0.94 | 6.23e-01 | 0.0077 (0.8%) |
| Residual | 2731.91 | 3625 | nan | nan | 0.4651 (46.5%) |

### OLS Regression (R-squared = 0.4040, Adj. R-squared = 0.4031)

| Predictor | Coefficient | SE | t | p |
|-----------|------------|-----|---|---|
| Intercept | -0.0095 | 0.0642 | -0.15 | 8.82e-01 |
| alpha | -0.0915 | 0.0765 | -1.20 | 2.32e-01 |
| sigma | -2.6194 | 0.0967 | -27.08 | 1.10e-147 |
| epsilon | -0.0864 | 0.1048 | -0.82 | 4.09e-01 |
| alpha:sigma | 0.1355 | 0.0987 | 1.37 | 1.70e-01 |
| alpha:epsilon | 0.0549 | 0.0790 | 0.70 | 4.87e-01 |
| sigma:epsilon | -0.3327 | 0.1579 | -2.11 | 3.52e-02 |

**Files:** `anova_table.csv`, `anova_convergence.csv`, `regression_summary.txt`

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
