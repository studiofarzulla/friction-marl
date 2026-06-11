# Learning Dynamics & Policy Convergence Report

**MARL 5x5x5 Factorial Experiment — GPU Run**
Generated: 2026-02-13

---

## 1. Experimental Summary

| Parameter | Values |
|-----------|--------|
| Conditions | 125 (5 alpha x 5 sigma x 5 epsilon) |
| Replications | 30 per condition (3,750 total) |
| Episodes | 1,000 per replication |
| Agents | 4 per replication |
| Algorithm | IQL (Independent Q-Learning) with target networks |
| Learning curve resolution | 20 windows of 50 episodes each |

**Overall mean reward:** -1.724 (SD = 1.252)

---

## 2. Learning Curve Analysis

### 2.1 Alpha Comparison

**Key finding:** At medium noise (sigma=0.6, epsilon=0.5), all alpha levels converge to similar final performance. The learning curves are nearly overlapping (fig1a), with cooperative (alpha=-0.8) and adversarial (alpha=0.8) agents reaching approximately -1.5 to -2.0 mean reward. The neutral condition (alpha=0.0) performs slightly worse on average (-2.486) compared to both cooperative (-1.631) and adversarial (-1.595).

This is a striking result: **friction (whether cooperative or adversarial) improves outcomes relative to the friction-free baseline**. The inverted-U pattern centered on alpha=0 suggests that any structural bias in agent preferences — toward alignment or opposition — provides a coordination signal that agents can exploit.

![Learning curves alpha comparison](fig1a_learning_curves_alpha_comparison.png)

### 2.2 Noise Effects

**Low noise (sigma=0.2):** All alpha levels learn quickly and achieve the best rewards. Preference spread is small, so agents' targets are similar and coordination is easy. Final rewards cluster around -0.4 to -0.6.

**High noise (sigma=1.0):** Dramatically slower learning and worse final performance. High sigma means agents have very different targets, making the multi-agent optimization problem fundamentally harder. Final rewards range from -2.0 to -4.0.

The full factorial grid (fig1d) shows epsilon (observation noise) has a secondary effect — it slightly degrades performance at high levels but the effect is modest compared to sigma.

### 2.3 Learning Speed

Episodes to reach 80% of final performance: remarkably uniform across all conditions at approximately 75-80 episodes. The initial jump from episode 0 to episode 50 accounts for most of the improvement, with subsequent training providing modest refinement. This suggests that IQL's epsilon-greedy exploration rapidly identifies reasonable policies, but fine-tuning under multi-agent dynamics is slow.

![Learning speed heatmap](fig1e_learning_speed_heatmap.png)

### 2.4 Learning Stability

Late-training stability (std of reward over last 250 episodes) is also remarkably uniform across conditions — around 0.04-0.06 for all alpha levels. Alpha=0 shows marginally higher instability (0.056 vs 0.043-0.050 for others). This means agents settle into stable policies regardless of friction level, but what they settle into varies dramatically.

![Learning stability](fig1f_learning_stability.png)

---

## 3. Convergence Analysis

### 3.1 Policy Convergence

**Policy convergence rate: 0.85%** — almost nothing converges by the policy-stability criterion. Out of 3,750 replications, only 32 achieve policy convergence before the maximum episode count. This is concentrated in high-sigma, high-epsilon conditions (which is counterintuitive until you realize these agents may converge to trivially random policies).

The near-zero convergence is not necessarily a failure — it reflects the multi-agent setting where agents continuously adjust to each other. True Nash equilibria in this 4-player resource allocation game are likely mixed strategies, and IQL doesn't naturally converge to mixed equilibria.

![Policy convergence heatmap](fig2a_convergence_heatmap.png)

### 3.2 Reward Convergence

**Reward convergence rate: 99.3%** — by contrast, almost all runs achieve stable reward levels. Using a 10% tolerance of final reward, 99.3% of replications plateau. This confirms that while policies keep drifting, the aggregate reward outcome stabilizes. Agents are in a dynamic equilibrium — constantly adjusting strategies but achieving similar collective outcomes.

Mean convergence episode for reward-converged runs varies by condition, with low-sigma conditions converging fastest (around 100-200 episodes) and high-sigma conditions taking 300-500 episodes.

![Reward convergence heatmap](fig2c_reward_convergence_heatmap.png)
![Reward convergence episode](fig2d_reward_convergence_episode.png)

### 3.3 Interpretation

The stark contrast between policy convergence (0.85%) and reward convergence (99.3%) is the most important finding here. It suggests:

1. **Policy cycling:** Agents continuously adapt to each other without reaching a fixed point
2. **Reward basins:** Despite policy cycling, the system reaches a reward attractor — the "effective outcome" is stable even when individual strategies are not
3. **This is exactly what the AoC friction framework predicts:** friction creates stable coordination outcomes without requiring convergence to a single equilibrium

---

## 4. Policy Space Analysis

### 4.1 PCA

The first 5 principal components explain 54.6% of policy variance (13.8%, 13.6%, 13.4%, 9.7%, 4.1%). The near-equal loading on the first three components suggests the policy space has approximately 3 independent dimensions of variation, which maps neatly onto our 3 experimental parameters.

**PCA colored by alpha** (fig3a left): No clear clustering by friction level. Policies from different alpha values are interleaved in the PCA space, meaning friction doesn't simply push agents into distinct behavioral regimes — it shapes the *distribution* within a shared policy space.

**PCA colored by reward** (fig3a right): High-reward policies (yellow/green) cluster toward the center of the PCA space, while low-reward policies (purple) are at the periphery. This suggests that good policies occupy a compact region of policy space, while bad policies are bad in diverse ways (a Tolstoy principle for MARL).

![PCA policy space](fig3a_pca_policy_space.png)

### 4.2 t-SNE

t-SNE reveals more local structure than PCA. While alpha levels still overlap substantially, there are visible micro-clusters where specific alpha values dominate. The reward coloring shows that t-SNE separates high-reward and low-reward policies more effectively than PCA.

![t-SNE policy space](fig3b_tsne_policy_space.png)

### 4.3 Policy Clustering

K-means elbow analysis identifies **k=4 distinct policy types**. Cluster composition by alpha (fig3d) shows that:

- Clusters are not alpha-pure — each cluster contains policies from all friction levels
- However, the *proportions* differ meaningfully across clusters
- This confirms that friction modulates the probability of reaching certain policy types rather than deterministically selecting them

![K-means elbow](fig3c_kmeans_elbow.png)
![Cluster composition](fig3d_cluster_composition.png)

---

## 5. Agent Specialization

### 5.1 The Alpha=0 Anomaly

The most striking finding: **agent reward variance peaks dramatically at alpha=0 (neutral), not at the extremes**. Mean agent variance:

| Alpha | Mean Agent Variance | Interpretation |
|-------|-------------------|----------------|
| -0.8 | 0.110 | Low — cooperative alignment equalizes rewards |
| -0.4 | 0.652 | Moderate |
| 0.0 | 2.841 | **Extremely high** — 28x higher than extremes |
| +0.4 | 0.636 | Moderate |
| +0.8 | 0.101 | Low — adversarial pressure also equalizes |

This is a key theoretical result: **friction (whether cooperative or adversarial) reduces inequality between agents**. The mechanism differs:

- **Cooperative friction (alpha < 0):** Agents' preferences are correlated, so they naturally coordinate toward shared targets. All agents benefit similarly.
- **Adversarial friction (alpha > 0):** Agents' preferences are anti-correlated, creating a zero-sum-like dynamic where exploitation is quickly punished by counteradaptation. The competitive pressure creates an equalizing force.
- **No friction (alpha = 0):** Preferences are uncorrelated (random). Some agents happen to have targets near the mean resource state, giving them a structural advantage. Without friction to create coordination or competition signals, these asymmetries persist.

![Agent specialization by alpha](fig4a_agent_specialization_alpha.png)

### 5.2 Sigma Amplifies Inequality

Agent variance increases monotonically with sigma (preference spread): from 0.108 at sigma=0.2 to 1.884 at sigma=1.0. Wider preference spread means greater potential for structural asymmetry between agents. This combines multiplicatively with the alpha=0 effect — the worst inequality occurs at (alpha=0, sigma=1.0).

![Specialization heatmap](fig4b_specialization_heatmap.png)

### 5.3 Significant Exploitation

Testing for persistent agent inequality (t-test between best and worst agents, p < 0.05):

| Alpha | Fraction with Significant Inequality |
|-------|--------------------------------------|
| -0.8 | 0% |
| -0.4 | 12% |
| 0.0 | 12% |
| +0.4 | 8% |
| +0.8 | 0% |

At the friction extremes, **zero conditions show significant exploitation**. This is strong evidence that friction acts as an equalizing mechanism. The moderate friction levels (-0.4, +0.4) and the neutral condition show occasional exploitation, but even there it's only 8-12% of conditions.

![Agent reward distributions](fig4c_agent_reward_distributions.png)

---

## 6. Reward Dynamics Over Training

### 6.1 Key Conditions

**Best condition (alpha=0.4, sigma=0.2, epsilon=0.0):** Rapid convergence to high reward (-0.43). Low sigma means easy coordination. Mild adversarial friction provides just enough competitive pressure to push agents toward efficient resource allocation without creating destructive competition. Tight IQR — very consistent across replications.

**Worst condition (alpha=0.0, sigma=1.0, epsilon=0.75):** Slow, noisy learning reaching only -4.51. High sigma creates fundamentally conflicting preferences, no friction signal to help coordinate, and high observation noise makes it harder to even observe the state accurately. Wide IQR — high variance across replications.

**Maximum variance condition (alpha=0.0, sigma=1.0, epsilon=0.75):** Same as worst — the conditions that produce the worst outcomes also produce the most variable outcomes. This confirms that the alpha=0, high-sigma regime is genuinely pathological for multi-agent coordination.

![Reward dynamics key conditions](fig5a_reward_dynamics_key_conditions.png)

### 6.2 Oscillation Patterns

Coefficient of variation (CV) analysis over sliding 5-window blocks reveals that oscillations are highest in early training (episodes 0-200) and decay over time for all conditions. High-noise conditions maintain higher CV throughout training. There's no evidence of late-stage oscillation or cycling — once agents reach their reward plateau, they stay there with low variance.

![Oscillation analysis](fig5b_oscillation_analysis.png)

### 6.3 Improvement Rate

The first derivative of the learning curve (reward improvement per window) shows that most learning happens in the first 50-100 episodes. After that, improvement rates are near zero for all conditions. Cooperative and adversarial agents show marginally faster initial improvement than neutral agents, but the difference is small.

![Improvement rate](fig5c_improvement_rate.png)

---

## 7. Key Takeaways for the AoC Paper

1. **Friction creates order:** Both cooperative and adversarial friction outperform the neutral baseline, confirming the central AoC thesis that friction dynamics are not merely obstacles but coordination mechanisms.

2. **The equalizing effect of friction:** Agent inequality (variance, Gini, significant exploitation) peaks at alpha=0 and drops dramatically at both extremes. This is the strongest empirical finding — friction acts as a structural equalizer regardless of sign.

3. **Reward convergence without policy convergence:** 99.3% of runs achieve stable rewards despite only 0.85% achieving stable policies. This is consistent with the friction framework's prediction of dynamic equilibria — stable outcomes through ongoing adaptation rather than fixed-point convergence.

4. **4 distinct policy types emerge** from K-means clustering, with friction modulating the probability distribution over these types rather than deterministically selecting them. This supports a stochastic view of friction's effects.

5. **Preference spread (sigma) is the dominant difficulty parameter.** It determines the fundamental hardness of multi-agent coordination more than friction level or observation noise.

6. **The worst-case scenario is the absence of friction.** Alpha=0 with high sigma produces both the worst mean outcomes and the highest inequality — the exact conditions where the AoC framework predicts coordination failure.

---

## 8. Figure Index

| Figure | Description | File |
|--------|-------------|------|
| 1a | Learning curves: alpha comparison (sigma=0.6, eps=0.5) | `fig1a_learning_curves_alpha_comparison` |
| 1b | Learning curves at low noise (sigma=0.2) | `fig1b_learning_curves_low_noise` |
| 1c | Learning curves at high noise (sigma=1.0) | `fig1c_learning_curves_high_noise` |
| 1d | Full factorial grid (3x5, sigma rows x alpha cols) | `fig1d_learning_curves_full_grid` |
| 1e | Learning speed heatmap (episodes to 80%) | `fig1e_learning_speed_heatmap` |
| 1f | Learning stability (late variance) | `fig1f_learning_stability` |
| 2a | Policy convergence heatmap (alpha x sigma) | `fig2a_convergence_heatmap` |
| 2b | Policy convergence by epsilon (5 panels) | `fig2b_convergence_by_epsilon` |
| 2c | Reward convergence heatmap | `fig2c_reward_convergence_heatmap` |
| 2d | Reward convergence episode heatmap | `fig2d_reward_convergence_episode` |
| 3a | PCA policy space (alpha + reward coloring) | `fig3a_pca_policy_space` |
| 3b | t-SNE policy space (alpha + reward coloring) | `fig3b_tsne_policy_space` |
| 3c | K-means elbow + clustering | `fig3c_kmeans_elbow` |
| 3d | Cluster composition by alpha | `fig3d_cluster_composition` |
| 4a | Agent specialization: variance + Gini by alpha | `fig4a_agent_specialization_alpha` |
| 4b | Specialization heatmap (alpha x sigma) | `fig4b_specialization_heatmap` |
| 4c | Per-agent reward distributions (6 conditions) | `fig4c_agent_reward_distributions` |
| 5a | Reward dynamics: best, worst, max-variance | `fig5a_reward_dynamics_key_conditions` |
| 5b | Oscillation analysis (CV over training) | `fig5b_oscillation_analysis` |
| 5c | Improvement rate over training | `fig5c_improvement_rate` |

All figures saved as 300 DPI PNG + PDF in `results/gpu_factorial/analysis/`.
