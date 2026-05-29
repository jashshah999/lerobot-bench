# lerobot-bench

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**A/B testing for robot learning policies.** Compare multiple policies on the same benchmark with statistical rigor.

```bash
pip install lerobot-bench
lerobot-bench compare user/policy_v1 user/policy_v2 --env pusht --n-episodes 50
```

```
┌──────────────────────────────────────────────────────────────┐
│                  Policy Comparison — pusht                     │
├────────────┬───────────┬──────────────┬────────────┬─────────┤
│ Policy     │ Success % │ 95% CI       │ Avg Reward │ Episodes│
├────────────┼───────────┼──────────────┼────────────┼─────────┤
│ policy_v1  │ 72.0%     │ [64.0, 80.0] │ 1.44 ± 0.3│ 50      │
│ policy_v2  │ 88.0% ★   │ [80.0, 94.0] │ 1.76 ± 0.2│ 50      │
└────────────┴───────────┴──────────────┴────────────┴─────────┘

Pairwise Significance Tests:
  policy_v1 vs policy_v2: p=0.0312, significant=YES, effect_size=0.521
```

## Why

Evaluating robot policies today means manually running `lerobot-eval` N times and eyeballing JSON files. There's no:
- Statistical significance testing (is A actually better than B, or just noise?)
- Confidence intervals (how much variance between seeds?)
- Standardized comparison format
- Regression detection for CI pipelines

lerobot-bench fills that gap.

## Install

```bash
pip install lerobot-bench
```

Or from source:
```bash
git clone https://github.com/jashshah999/lerobot-bench.git
cd lerobot-bench
pip install -e ".[dev]"
```

## Usage

### Compare policies

```bash
# Compare two HuggingFace Hub policies
lerobot-bench compare user/policy_a user/policy_b --env pusht --n-episodes 50

# Multiple seeds for statistical power
lerobot-bench compare policy_a policy_b --env libero --task libero_10 --seeds 1000,2000,3000

# Export results
lerobot-bench compare policy_a policy_b --env pusht -o results/ --format markdown
```

### Analyze existing results

Already have `eval_info.json` files from previous runs? Skip re-evaluation:

```bash
lerobot-bench analyze eval_run_a/eval_info.json eval_run_b/eval_info.json
```

### Checkpoint sweep

Find the best checkpoint along a training run:

```bash
lerobot-bench sweep ./outputs/train --checkpoints 'step_*' --env pusht --n-episodes 20
```

### Regression detection (for CI)

Returns exit code 1 if the candidate is worse than baseline:

```bash
lerobot-bench regression user/policy_v1 user/policy_v2 --env pusht --threshold 0.05
```

Use in GitHub Actions:
```yaml
- name: Check for regression
  run: lerobot-bench regression $BASELINE $CANDIDATE --env pusht
```

## What it does

1. **Orchestrates** — runs `lerobot-eval` for each policy x seed combination
2. **Aggregates** — collects per-episode success/reward across seeds
3. **Statistics** — bootstrap confidence intervals, Mann-Whitney U test, Cohen's d effect size
4. **Reports** — rich terminal tables, markdown, CSV, JSON export
5. **Detects regressions** — CI-friendly exit codes

## Metrics

| Metric | Description |
|--------|-------------|
| Success % | Mean success rate across episodes and seeds |
| 95% CI | Bootstrap confidence interval on success rate |
| Avg Reward | Mean cumulative reward |
| p-value | Mann-Whitney U test between policy pairs |
| Effect size | Cohen's d (small: 0.2, medium: 0.5, large: 0.8) |

## Python API

```python
from lerobot_bench import run_comparison, render_table

results = run_comparison(
    policy_paths=["user/policy_a", "user/policy_b"],
    env_config="pusht",
    task=None,
    n_episodes=50,
    seeds=[1000, 2000, 3000],
)

render_table(results)
```

## How it compares

| | Manual eval | vla-evaluation-harness | **lerobot-bench** |
|---|---|---|---|
| Multi-policy comparison | Manual JSON diff | No (single policy focus) | **Yes** |
| Statistical significance | No | No | **Yes (p-values, CI)** |
| Confidence intervals | No | No | **Yes (bootstrap)** |
| Checkpoint sweeping | Manual | No | **Yes** |
| Regression detection | No | No | **Yes (CI-friendly)** |
| Lightweight install | N/A | Heavy (Docker) | **Minimal deps** |

## License

MIT
