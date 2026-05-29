"""Run policy evaluations and collect results."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class LeRobotNotInstalledError(RuntimeError):
    """Raised when lerobot is not available for running evaluations."""

    def __init__(self):
        super().__init__(
            "lerobot is not installed. Install it with:\n"
            "  pip install lerobot\n\n"
            "Alternatively, use 'lerobot-bench analyze' to compare existing eval_info.json files "
            "without needing lerobot installed."
        )


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


def _check_lerobot_available():
    """Check if lerobot is importable, raise helpful error if not."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("lerobot")
        if spec is None:
            raise LeRobotNotInstalledError()
    except ModuleNotFoundError:
        raise LeRobotNotInstalledError()


def _run_lerobot_eval(policy_path, env_config, task, n_episodes, seed, device, batch_size, output_dir,
                      timeout=3600):
    """Run a single lerobot-eval invocation and return parsed results."""
    _check_lerobot_available()

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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        eval_time = time.time() - t0
        return None, eval_time, f"Evaluation timed out after {timeout}s"
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
        try:
            result = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, eval_time, f"Evaluation timed out after {timeout}s"
        if result.returncode != 0:
            return None, eval_time, result.stderr[-500:] if result.stderr else "Unknown error"

    # Parse eval_info.json
    eval_info_path = os.path.join(output_dir, "eval_info.json")
    if not os.path.exists(eval_info_path):
        # Try to find it in subdirectories
        for root, dirs, files in os.walk(output_dir):
            if "eval_info.json" in files:
                eval_info_path = os.path.join(root, "eval_info.json")
                break

    if os.path.exists(eval_info_path):
        try:
            with open(eval_info_path) as f:
                data = json.load(f)
            return data, eval_time, None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return None, eval_time, f"Failed to parse eval_info.json: {e}"
    return None, eval_time, "eval_info.json not found"


def _parse_eval_info(eval_info):
    """Extract metrics from lerobot eval_info.json format."""
    metrics = {}

    if "overall" in eval_info:
        # Multi-task lerobot format
        overall = eval_info["overall"]
        pc = overall.get("pc_success", 0.0)
        metrics["success_rate"] = pc / 100.0 if pc > 1.0 else pc
        metrics["avg_reward"] = overall.get("avg_sum_reward", 0.0)
        metrics["eval_s"] = overall.get("eval_s", 0.0)
    elif "aggregated" in eval_info:
        agg = eval_info["aggregated"]
        pc = agg.get("pc_success", agg.get("avg_max_reward", 0.0))
        metrics["success_rate"] = pc / 100.0 if pc > 1.0 else pc
        metrics["avg_reward"] = agg.get("avg_sum_reward", 0.0)
        metrics["eval_s"] = agg.get("eval_s", 0.0)
    elif "pc_success" in eval_info:
        pc = eval_info["pc_success"]
        metrics["success_rate"] = pc / 100.0 if pc > 1.0 else pc
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


def run_comparison(policy_paths, env_config, task, n_episodes, seeds, device="cuda", batch_size=1,
                   timeout=3600):
    """Run comparison across multiple policies and seeds."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.console import Console

    console = Console()

    # Fail fast if lerobot isn't available
    _check_lerobot_available()

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
                    policy_path, env_config, task, n_episodes, seed, device, batch_size, output_dir,
                    timeout=timeout,
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
