"""Render comparison results as tables and export to files."""

import json
import os

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from lerobot_bench.stats import compute_statistics, significance_test


def render_table(comparison_result, markdown=False):
    """Render comparison results as a rich table or markdown."""
    policies = comparison_result.policies

    if markdown:
        _render_markdown(policies)
    else:
        _render_rich(policies, comparison_result)


def _render_rich(policies, comparison_result):
    """Rich terminal table."""
    console = Console()

    table = Table(title=f"Policy Comparison — {comparison_result.task or comparison_result.env_config}")
    table.add_column("Policy", style="cyan", no_wrap=True)
    table.add_column("Success %", justify="right", style="green")
    table.add_column("95% CI", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Episodes", justify="right")
    table.add_column("Eval Time", justify="right", style="dim")

    best_success = -1
    best_idx = 0
    for i, p in enumerate(policies):
        mean_sr = np.mean(p.success_rates) if p.success_rates else 0.0
        if mean_sr > best_success:
            best_success = mean_sr
            best_idx = i

    for i, policy in enumerate(policies):
        sr_stats = compute_statistics(policy.success_rates)
        reward_stats = compute_statistics(policy.rewards)

        sr_str = f"{sr_stats['mean']*100:.1f}%"
        if i == best_idx and len(policies) > 1:
            sr_str = f"[bold]{sr_str} ★[/bold]"

        ci_str = f"[{sr_stats['ci_low']*100:.1f}, {sr_stats['ci_high']*100:.1f}]"
        reward_str = f"{reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}"
        n_eps = len(policy.episodes)
        time_str = f"{policy.eval_time_s:.0f}s"

        table.add_row(policy.name, sr_str, ci_str, reward_str, str(n_eps), time_str)

    console.print(table)

    # Significance tests (pairwise)
    if len(policies) >= 2:
        console.print("\n[bold]Pairwise Significance Tests:[/bold]")
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                if not policies[i].success_rates or not policies[j].success_rates:
                    continue
                # Use per-episode success for significance
                ep_a = [float(ep["success"]) for ep in policies[i].episodes]
                ep_b = [float(ep["success"]) for ep in policies[j].episodes]
                if not ep_a or not ep_b:
                    continue
                test = significance_test(ep_a, ep_b)
                sig_str = "[green]YES[/green]" if test["significant"] else "[red]NO[/red]"
                console.print(
                    f"  {policies[i].name} vs {policies[j].name}: "
                    f"p={test['p_value']:.4f}, significant={sig_str}, "
                    f"effect_size={test['effect_size']:.3f}"
                )


def _render_markdown(policies):
    """Print markdown table to stdout."""
    print("| Policy | Success % | 95% CI | Avg Reward | Episodes |")
    print("|--------|-----------|--------|------------|----------|")
    for policy in policies:
        sr_stats = compute_statistics(policy.success_rates)
        reward_stats = compute_statistics(policy.rewards)
        ci_str = f"[{sr_stats['ci_low']*100:.1f}, {sr_stats['ci_high']*100:.1f}]"
        print(
            f"| {policy.name} | {sr_stats['mean']*100:.1f}% | {ci_str} | "
            f"{reward_stats['mean']:.2f} ± {reward_stats['std']:.2f} | {len(policy.episodes)} |"
        )


def export_results(comparison_result, output_dir, formats=None):
    """Export results to JSON and/or CSV."""
    if formats is None:
        formats = ["json", "csv"]

    data = {
        "env": comparison_result.env_config,
        "task": comparison_result.task,
        "n_episodes": comparison_result.n_episodes,
        "seeds": comparison_result.seeds,
        "policies": [],
    }

    rows = []
    for policy in comparison_result.policies:
        sr_stats = compute_statistics(policy.success_rates)
        reward_stats = compute_statistics(policy.rewards)

        policy_data = {
            "name": policy.name,
            "path": policy.path,
            "success_rate": sr_stats,
            "reward": reward_stats,
            "n_episodes": len(policy.episodes),
            "eval_time_s": policy.eval_time_s,
        }
        data["policies"].append(policy_data)

        rows.append({
            "policy": policy.name,
            "success_mean": sr_stats["mean"],
            "success_std": sr_stats["std"],
            "success_ci_low": sr_stats["ci_low"],
            "success_ci_high": sr_stats["ci_high"],
            "reward_mean": reward_stats["mean"],
            "reward_std": reward_stats["std"],
            "n_episodes": len(policy.episodes),
            "eval_time_s": policy.eval_time_s,
        })

    if "json" in formats:
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(data, f, indent=2)

    if "csv" in formats:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)
