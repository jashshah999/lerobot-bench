"""CLI for lerobot-bench."""

import click
import json
import os
import sys
from pathlib import Path


@click.group()
@click.version_option()
def main():
    """Compare robot learning policies with statistical rigor."""
    pass


@main.command()
@click.argument("policies", nargs=-1, required=True)
@click.option("--env", type=str, help="Environment config (e.g. 'lerobot/configs/env/libero.yaml')")
@click.option("--task", type=str, help="Task name within the environment")
@click.option("--n-episodes", type=int, default=50, help="Episodes per policy per seed")
@click.option("--seeds", type=str, default="1000", help="Comma-separated seeds")
@click.option("--output", "-o", type=str, default=None, help="Output directory for results")
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "json", "markdown"]), default="table")
@click.option("--device", type=str, default="cuda", help="Device for inference")
@click.option("--batch-size", type=int, default=1, help="Batch size for vectorized envs")
def compare(policies, env, task, n_episodes, seeds, output, fmt, device, batch_size):
    """Compare multiple policies on the same environment.

    \b
    Examples:
        lerobot-bench compare user/policy_a user/policy_b --env libero --task libero_10
        lerobot-bench compare ./checkpoint_10k ./checkpoint_50k --env pusht
        lerobot-bench compare policy_a policy_b --seeds 1000,2000,3000
    """
    from lerobot_bench.runner import run_comparison
    from lerobot_bench.report import render_table, export_results

    seed_list = [int(s) for s in seeds.split(",")]

    results = run_comparison(
        policy_paths=list(policies),
        env_config=env,
        task=task,
        n_episodes=n_episodes,
        seeds=seed_list,
        device=device,
        batch_size=batch_size,
    )

    if fmt == "table":
        render_table(results)
    elif fmt == "markdown":
        render_table(results, markdown=True)

    if output:
        os.makedirs(output, exist_ok=True)
        export_results(results, output, formats=["json", "csv"])
        click.echo(f"\nResults saved to {output}/")


@main.command()
@click.argument("result_files", nargs=-1, required=True)
@click.option("--format", "fmt", type=click.Choice(["table", "csv", "markdown"]), default="table")
@click.option("--significance", type=float, default=0.05, help="Significance level for tests")
def analyze(result_files, fmt, significance):
    """Analyze and compare existing eval_info.json files.

    \b
    Examples:
        lerobot-bench analyze eval_policy_a/eval_info.json eval_policy_b/eval_info.json
        lerobot-bench analyze results/*.json --format markdown
    """
    from lerobot_bench.analyzer import analyze_results
    from lerobot_bench.report import render_table

    results = analyze_results(list(result_files), significance_level=significance)
    render_table(results, markdown=(fmt == "markdown"))


@main.command()
@click.argument("policy_path", type=str)
@click.option("--checkpoints", type=str, help="Glob pattern for checkpoints (e.g. 'checkpoints/step_*')")
@click.option("--env", type=str, help="Environment config")
@click.option("--task", type=str, help="Task name")
@click.option("--n-episodes", type=int, default=20)
@click.option("--output", "-o", type=str, default=None)
def sweep(policy_path, checkpoints, env, task, n_episodes, output):
    """Sweep across checkpoints to find the best one.

    \b
    Examples:
        lerobot-bench sweep ./outputs/train --checkpoints 'step_*' --env pusht
        lerobot-bench sweep user/model --checkpoints 'checkpoint-*' --env libero
    """
    import glob
    from lerobot_bench.runner import run_comparison
    from lerobot_bench.report import render_table, export_results

    if checkpoints:
        pattern = os.path.join(policy_path, checkpoints)
        paths = sorted(glob.glob(pattern))
        if not paths:
            click.echo(f"No checkpoints found matching: {pattern}", err=True)
            sys.exit(1)
    else:
        click.echo("--checkpoints pattern required for sweep", err=True)
        sys.exit(1)

    click.echo(f"Found {len(paths)} checkpoints")
    results = run_comparison(
        policy_paths=paths,
        env_config=env,
        task=task,
        n_episodes=n_episodes,
        seeds=[1000],
        device="cuda",
    )

    render_table(results)
    if output:
        os.makedirs(output, exist_ok=True)
        export_results(results, output, formats=["json", "csv"])


@main.command()
@click.argument("baseline", type=str)
@click.argument("candidate", type=str)
@click.option("--threshold", type=float, default=0.05, help="Regression threshold (fraction)")
@click.option("--env", type=str, help="Environment config")
@click.option("--task", type=str, help="Task name")
@click.option("--n-episodes", type=int, default=50)
def regression(baseline, candidate, threshold, env, task, n_episodes):
    """Check if a candidate policy regresses against a baseline.

    Exit code 0 = no regression, 1 = regression detected.
    Useful in CI pipelines.

    \b
    Examples:
        lerobot-bench regression user/policy_v1 user/policy_v2 --env pusht
    """
    from lerobot_bench.runner import run_comparison
    from lerobot_bench.stats import detect_regression

    results = run_comparison(
        policy_paths=[baseline, candidate],
        env_config=env,
        task=task,
        n_episodes=n_episodes,
        seeds=[1000, 2000, 3000],
        device="cuda",
    )

    regressed = detect_regression(results, threshold=threshold)
    if regressed:
        click.echo(f"REGRESSION DETECTED: candidate is >{threshold*100}% worse than baseline")
        sys.exit(1)
    else:
        click.echo("No regression detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
