# marl-rom

Multi-Agent Reinforcement Learning simulation for testing a coordination friction framework.

## Overview
This codebase simulates a multi-agent resource allocation environment and evaluates whether coordination failure correlates with the theoretical friction function:

F = σ × (1 + ε) / (1 + α)

It runs a 5×5×5 factorial design over alignment (α), stakes (σ), and observation entropy (ε), with replications per condition, trains independent Q-learning agents, and computes friction proxies plus regression analyses.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_experiments.py --output-dir ./results
```

## CLI

```bash
python run_experiments.py \
  --n-agents 4 \
  --n-resources 3 \
  --n-replications 30 \
  --n-episodes 10000 \
  --output-dir ./results \
  --seed 123
```

Outputs:
- Raw results: CSV in `results/`
- Analysis tables and plots in `results/analysis/`

## Notes
- Default settings are computationally heavy (125 conditions × 30 replications × 10,000 episodes). Consider reducing episodes or replications for quick tests.
- The environment uses continuous resource levels, with agent actions from {-1, 0, +1} per resource.

## Project Layout

```
friction_marl/
  envs/
  agents/
  experiments/
  utils/
run_experiments.py
```
