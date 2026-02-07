# Friction-MARL

**Multi-Agent Reinforcement Learning with Friction Dynamics**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code companion to: [Axiom of Consent: Friction Dynamics in Multi-Agent Coordination](https://arxiv.org/abs/2601.06692) (DAI-2601)

## Overview

Multi-agent reinforcement learning simulation for testing coordination friction frameworks. The codebase simulates a multi-agent resource allocation environment and evaluates whether coordination failure correlates with the theoretical friction function:

```
F = sigma * (1 + epsilon) / (1 + alpha)
```

It runs a 5x5x5 factorial design over alignment (alpha), stakes (sigma), and observation entropy (epsilon), with replications per condition, trains independent Q-learning agents, and computes friction proxies plus regression analyses.

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

Outputs: Raw results CSV in `results/`, analysis tables and plots in `results/analysis/`.

Default settings are computationally heavy (125 conditions x 30 replications x 10,000 episodes). Reduce episodes or replications for quick tests.

## Project Layout

```
friction-marl/
├── friction_marl/
│   ├── envs/          # Resource allocation environments
│   ├── agents/        # Q-learning agent implementations
│   ├── experiments/   # Experiment configurations
│   └── utils/         # Analysis and plotting utilities
├── run_experiments.py # Main entry point
├── requirements.txt   # Dependencies
└── pyproject.toml     # Package configuration
```

## Related Papers

- Axiom of Consent (DAI-2601): [arXiv:2601.06692](https://arxiv.org/abs/2601.06692)
- ROM (DAI-2503): [arXiv:2601.06363](https://arxiv.org/abs/2601.06363)

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai)
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)

## License

MIT License
