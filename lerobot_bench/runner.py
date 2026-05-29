"""Run policy evaluations and collect results."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PolicyResult:
    name: str
    path: str
    episodes: list = field(default_factory=list)
    success_rates: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    inference_times_ms: list = field(default_factory=list)
    eval_time_s: float = 0.0
    raw_results: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    policies: list  # list of PolicyResult
    env_config: str = ""
    task: str = ""
    n_episodes: int = 0
    seeds: list = field(default_factory=list)


def _run_lerobot_eval(policy_path, env_config, task, n_episodes, seed, device, batch_size, output_dir):
    """Run a single lerobot-eval invocation and return parsed results."""
    cmd = [
        sys.executable, "-m", "lerobot.scripts.eval",
        f"--policy.path={policy_path}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
    ]
    if env_config:
        cmd.append(f"--env.type={env_config}")
    if task:
        cmd.append(f"--env.task={task}")
    if device:
        cmd.append(f"--device={device}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    eval_time = time.time() - t0

    if result.returncode != 0:
        # Try alternative eval command format
        cmd_alt = [
            sys.executable, "-m", "lerobot_eval",
            f"--policy.path={policy_path}",
            f"--eval.n_episodes={n_episodes}",
            f"--seed={seed}",
            f"--output_dir={output_dir}",
        ]
        if env_config:
            cmd_alt.append(f"--env.type={env_config}")
        result = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            return None, eval_time, result.stderr[-500:]

    # Parse eval_info.json
    eval_info_path = os.path.join(output_dir, "eval_info.json")
    if not os.path.exists(eval_info_path):
        # Try to find it in subdirectories
        for root, dirs, files in os.walk(output_dir):
            if "eval_info.json" in files:
                eval_info_path = os.path.join(root, "eval_info.json")
                break

    if os.path.exists(eval_info_path):
        with open(eval_info_path) as f:
            return json.load(f), eval_time, None
    return None, eval_time, "eval_info.json not found"


def _parse_eval_info(eval_info):
    """Extract metrics from lerobot eval_info.json format."""
    metrics = {}

    if "aggregated" in eval_info:
        agg = eval_info["aggregated"]
        metrics["success_rate"] = agg.get("pc_success", agg.get("avg_max_reward", 0.0))
        metrics["avg_reward"] = agg.get("avg_sum_reward", 0.0)
        metrics["eval_s"] = agg.get("eval_s", 0.0)
    elif "pc_success" in eval_info:
        metrics["success_rate"] = eval_info["pc_success"]
        metrics["avg_reward"] = eval_info.get("avg_sum_reward", 0.0)
        metrics["eval_s"] = eval_info.get("eval_s", 0.0)

    # Per-episode data
    episodes = []
    if "per_episode" in eval_info:
        for ep in eval_info["per_episode"]:
            episodes.append({
                "success": ep.get("success", ep.get("max_reward", 0.0) >= 1.0),
                "reward": ep.get("sum_reward", 0.0),
                "steps": ep.get("steps", 0),
            })
    elif "episodes" in eval_info:
        for ep in eval_info["episodes"]:
            episodes.append({
                "success": ep.get("success", False),
                "reward": ep.get("reward", 0.0),
                "steps": ep.get("length", 0),
            })

    metrics["episodes"] = episodes
    return metrics


def run_comparison(policy_paths, env_config, task, n_episodes, seeds, device="cuda", batch_size=1):
    """Run comparison across multiple policies and seeds."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.console import Console

    console = Console()
    results = []

    total_runs = len(policy_paths) * len(seeds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Evaluating policies...", total=total_runs)

        for policy_path in policy_paths:
            policy_name = Path(policy_path).name or policy_path.split("/")[-1]
            policy_result = PolicyResult(name=policy_name, path=policy_path)

            for seed in seeds:
                progress.update(task_id, description=f"{policy_name} (seed={seed})")

                output_dir = f"/tmp/lerobot_bench/{policy_name}/seed_{seed}"
                os.makedirs(output_dir, exist_ok=True)

                eval_info, eval_time, error = _run_lerobot_eval(
                    policy_path, env_config, task, n_episodes, seed, device, batch_size, output_dir
                )

                if eval_info is None:
                    console.print(f"[red]FAILED: {policy_name} seed={seed}: {error}[/red]")
                    progress.advance(task_id)
                    continue

                metrics = _parse_eval_info(eval_info)
                policy_result.success_rates.append(metrics.get("success_rate", 0.0))
                policy_result.rewards.append(metrics.get("avg_reward", 0.0))
                policy_result.episodes.extend(metrics.get("episodes", []))
                policy_result.eval_time_s += eval_time
                policy_result.raw_results[f"seed_{seed}"] = eval_info

                progress.advance(task_id)

            results.append(policy_result)

    return ComparisonResult(
        policies=results,
        env_config=env_config or "",
        task=task or "",
        n_episodes=n_episodes,
        seeds=seeds,
    )
