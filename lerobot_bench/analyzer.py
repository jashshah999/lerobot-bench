"""Analyze existing eval_info.json files without re-running evaluations."""

import json
import os
from pathlib import Path

import numpy as np

from lerobot_bench.runner import PolicyResult, ComparisonResult


def load_eval_info(filepath):
    """Load and parse an eval_info.json file."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def analyze_results(result_files, significance_level=0.05):
    """Load multiple eval_info.json files and build a ComparisonResult."""
    policies = []

    for filepath in result_files:
        if not os.path.exists(filepath):
            continue

        data = load_eval_info(filepath)
        name = Path(filepath).parent.name or Path(filepath).stem

        policy = PolicyResult(name=name, path=filepath)

        # Extract success rate
        if "aggregated" in data:
            policy.success_rates.append(data["aggregated"].get("pc_success", 0.0))
            policy.rewards.append(data["aggregated"].get("avg_sum_reward", 0.0))
        elif "pc_success" in data:
            policy.success_rates.append(data["pc_success"])
            policy.rewards.append(data.get("avg_sum_reward", 0.0))

        # Per-episode data for significance testing
        if "per_episode" in data:
            for ep in data["per_episode"]:
                policy.episodes.append({
                    "success": ep.get("success", ep.get("max_reward", 0) >= 1.0),
                    "reward": ep.get("sum_reward", 0.0),
                    "steps": ep.get("steps", 0),
                })
        elif "episodes" in data:
            for ep in data["episodes"]:
                policy.episodes.append({
                    "success": ep.get("success", False),
                    "reward": ep.get("reward", 0.0),
                    "steps": ep.get("length", 0),
                })

        # If we have per-episode data, compute success rate from it
        if policy.episodes and not policy.success_rates:
            sr = np.mean([float(ep["success"]) for ep in policy.episodes])
            policy.success_rates.append(sr)

        policies.append(policy)

    return ComparisonResult(policies=policies)
