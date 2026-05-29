"""Analyze existing eval_info.json files without re-running evaluations."""

import json
import os
from pathlib import Path

import numpy as np

from lerobot_bench.runner import PolicyResult, ComparisonResult


def load_eval_info(filepath):
    """Load and parse an eval_info.json file.

    Raises ValueError if the file cannot be parsed as JSON.
    """
    with open(filepath) as f:
        try:
            data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to parse {filepath}: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {filepath}, got {type(data).__name__}")
    return data


def analyze_results(result_files, significance_level=0.05):
    """Load multiple eval_info.json files and build a ComparisonResult."""
    policies = []

    for filepath in result_files:
        if not os.path.exists(filepath):
            continue

        try:
            data = load_eval_info(filepath)
        except ValueError:
            # Skip files that can't be parsed
            continue
        name = Path(filepath).parent.name or Path(filepath).stem

        policy = PolicyResult(name=name, path=filepath)

        # Extract success rate — handle both single-task and multi-task formats
        # Multi-task format: {"per_task": {...}, "overall": {...}}
        # Single-task format: {"aggregated": {...}, "per_episode": [...]}
        if "overall" in data:
            # Multi-task lerobot format
            overall = data["overall"]
            pc = overall.get("pc_success", 0.0)
            # lerobot stores pc_success as percentage (0-100)
            policy.success_rates.append(pc / 100.0 if pc > 1.0 else pc)
            policy.rewards.append(overall.get("avg_sum_reward", 0.0))
        elif "aggregated" in data:
            agg = data["aggregated"]
            pc = agg.get("pc_success", 0.0)
            # lerobot stores pc_success as percentage (0-100)
            policy.success_rates.append(pc / 100.0 if pc > 1.0 else pc)
            policy.rewards.append(agg.get("avg_sum_reward", 0.0))
        elif "pc_success" in data:
            pc = data["pc_success"]
            policy.success_rates.append(pc / 100.0 if pc > 1.0 else pc)
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
