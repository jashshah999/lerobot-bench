"""Integration and edge-case tests for lerobot-bench."""

import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from lerobot_bench.analyzer import analyze_results, load_eval_info
from lerobot_bench.runner import (
    ComparisonResult,
    LeRobotNotInstalledError,
    PolicyResult,
    _check_lerobot_available,
    _parse_eval_info,
)
from lerobot_bench.stats import compute_statistics, significance_test


# ============================================================
# Fixtures: realistic lerobot eval_info.json data
# ============================================================

def _make_lerobot_eval_info(n_episodes=50, success_rate=0.8, seed_start=1000):
    """Create eval_info.json matching lerobot's actual output format.

    This matches the format from lerobot/scripts/lerobot_eval.py:eval_policy()
    """
    successes = [i < int(success_rate * n_episodes) for i in range(n_episodes)]
    np.random.seed(42)
    sum_rewards = [float(s) + np.random.uniform(-0.1, 0.1) for s in successes]
    max_rewards = [float(s) for s in successes]

    return {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_rewards[i],
                "max_reward": max_rewards[i],
                "success": successes[i],
                "seed": seed_start + i,
            }
            for i in range(n_episodes)
        ],
        "aggregated": {
            "avg_sum_reward": float(np.mean(sum_rewards)),
            "avg_max_reward": float(np.mean(max_rewards)),
            "pc_success": float(np.mean(successes) * 100),
            "eval_s": 120.5,
            "eval_ep_s": 120.5 / n_episodes,
        },
    }


def _make_lerobot_multitask_eval_info():
    """Create eval_info.json matching lerobot's multi-task format."""
    return {
        "per_task": {
            "task_0": {
                "avg_sum_reward": 0.9,
                "avg_max_reward": 0.9,
                "pc_success": 90.0,
                "n_episodes": 10,
                "video_paths": [],
            },
            "task_1": {
                "avg_sum_reward": 0.6,
                "avg_max_reward": 0.6,
                "pc_success": 60.0,
                "n_episodes": 10,
                "video_paths": [],
            },
        },
        "per_group": {},
        "overall": {
            "avg_sum_reward": 0.75,
            "avg_max_reward": 0.75,
            "pc_success": 75.0,
            "n_episodes": 20,
            "eval_s": 60.0,
            "eval_ep_s": 3.0,
            "video_paths": [],
        },
    }


# ============================================================
# Edge cases: all fail, all succeed, single episode
# ============================================================

class TestEdgeCases:
    def test_all_episodes_fail(self):
        """0% success rate."""
        data = _make_lerobot_eval_info(n_episodes=20, success_rate=0.0)
        assert data["aggregated"]["pc_success"] == 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy_fail", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = analyze_results([path])
            assert len(result.policies) == 1
            assert result.policies[0].success_rates[0] == 0.0
            assert all(ep["success"] is False for ep in result.policies[0].episodes)

    def test_all_episodes_succeed(self):
        """100% success rate."""
        data = _make_lerobot_eval_info(n_episodes=20, success_rate=1.0)
        assert data["aggregated"]["pc_success"] == 100.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "policy_perfect", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = analyze_results([path])
            assert len(result.policies) == 1
            assert result.policies[0].success_rates[0] == 1.0
            assert all(ep["success"] is True for ep in result.policies[0].episodes)

    def test_single_episode(self):
        """Only 1 episode in the eval."""
        data = _make_lerobot_eval_info(n_episodes=1, success_rate=1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "single_ep", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = analyze_results([path])
            assert len(result.policies) == 1
            assert len(result.policies[0].episodes) == 1

    def test_single_episode_statistics(self):
        """Statistics with single value should not crash."""
        stats = compute_statistics([1.0])
        assert stats["mean"] == 1.0
        assert stats["n"] == 1
        assert stats["std"] == 0.0

    def test_two_values_statistics(self):
        """Statistics with 2 values uses fallback CI."""
        stats = compute_statistics([0.0, 1.0])
        assert stats["mean"] == 0.5
        assert stats["n"] == 2
        assert stats["ci_low"] < stats["ci_high"]


# ============================================================
# Malformed / unexpected JSON formats
# ============================================================

class TestMalformedInput:
    def test_invalid_json(self):
        """Completely invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write("this is not json {{{")

            result = analyze_results([path])
            assert len(result.policies) == 0  # gracefully skipped

    def test_json_array_instead_of_object(self):
        """JSON is valid but not an object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "array", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump([1, 2, 3], f)

            result = analyze_results([path])
            assert len(result.policies) == 0  # gracefully skipped

    def test_empty_object(self):
        """JSON is {} with no recognized keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump({}, f)

            result = analyze_results([path])
            # Should still create a policy entry but with no data
            assert len(result.policies) == 1
            assert result.policies[0].success_rates == []

    def test_missing_per_episode_keys(self):
        """per_episode entries missing expected keys."""
        data = {
            "aggregated": {"pc_success": 50.0, "avg_sum_reward": 1.0},
            "per_episode": [
                {"episode_ix": 0},  # no success, no reward
                {"episode_ix": 1, "max_reward": 1.0},  # infer success from max_reward
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "partial", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = analyze_results([path])
            assert len(result.policies) == 1
            assert len(result.policies[0].episodes) == 2

    def test_load_eval_info_raises_on_bad_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as f:
                f.write("not json")
            with pytest.raises(ValueError, match="Failed to parse"):
                load_eval_info(path)

    def test_load_eval_info_raises_on_non_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "list.json")
            with open(path, "w") as f:
                json.dump([1, 2], f)
            with pytest.raises(ValueError, match="Expected a JSON object"):
                load_eval_info(path)


# ============================================================
# Confidence intervals with extreme variance
# ============================================================

class TestConfidenceIntervals:
    def test_zero_variance(self):
        """All identical values."""
        stats = compute_statistics([0.5] * 100)
        assert stats["mean"] == 0.5
        assert stats["ci_low"] == pytest.approx(0.5, abs=0.01)
        assert stats["ci_high"] == pytest.approx(0.5, abs=0.01)

    def test_high_variance(self):
        """Alternating 0 and 1 — max variance for binary outcomes."""
        values = [0.0, 1.0] * 50
        stats = compute_statistics(values)
        assert stats["mean"] == pytest.approx(0.5, abs=0.01)
        # CI should be wide
        assert stats["ci_high"] - stats["ci_low"] > 0.1

    def test_ci_contains_mean(self):
        """CI should always contain the mean."""
        rng = np.random.default_rng(123)
        values = rng.normal(0.7, 0.2, 30).tolist()
        stats = compute_statistics(values)
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


# ============================================================
# Runner robustness
# ============================================================

class TestRunnerRobustness:
    def test_lerobot_not_installed_error(self):
        """Check graceful error when lerobot is not available."""
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(LeRobotNotInstalledError, match="lerobot is not installed"):
                _check_lerobot_available()

    def test_lerobot_not_installed_suggests_analyze(self):
        """Error message should suggest using analyze mode."""
        err = LeRobotNotInstalledError()
        assert "analyze" in str(err)

    def test_parse_eval_info_single_task(self):
        """Parse the standard single-task lerobot format."""
        data = _make_lerobot_eval_info(n_episodes=10, success_rate=0.7)
        metrics = _parse_eval_info(data)
        assert metrics["success_rate"] == pytest.approx(0.7, abs=0.01)
        assert len(metrics["episodes"]) == 10

    def test_parse_eval_info_multi_task(self):
        """Parse the multi-task lerobot format."""
        data = _make_lerobot_multitask_eval_info()
        metrics = _parse_eval_info(data)
        assert metrics["success_rate"] == pytest.approx(0.75, abs=0.01)

    def test_parse_eval_info_empty(self):
        """Parse an empty dict without crashing."""
        metrics = _parse_eval_info({})
        assert metrics.get("episodes", []) == []


# ============================================================
# Integration: lerobot's actual format through analyze_results
# ============================================================

class TestLeRobotIntegration:
    def test_real_format_single_task(self):
        """Full integration test with lerobot's exact single-task output."""
        data = _make_lerobot_eval_info(n_episodes=50, success_rate=0.72)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "act_policy", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = analyze_results([path])
            policy = result.policies[0]
            assert policy.name == "act_policy"
            assert policy.success_rates[0] == pytest.approx(0.72, abs=0.01)
            assert len(policy.episodes) == 50
            # Verify episodes have correct structure
            ep = policy.episodes[0]
            assert "success" in ep
            assert "reward" in ep

    def test_real_format_multi_task(self):
        """Full integration test with lerobot's multi-task output."""
        data = _make_lerobot_multitask_eval_info()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "diffusion_policy", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = analyze_results([path])
            policy = result.policies[0]
            assert policy.name == "diffusion_policy"
            assert policy.success_rates[0] == pytest.approx(0.75, abs=0.01)

    def test_compare_two_policies_real_format(self):
        """Compare two policies with real lerobot format end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for name, sr in [("act_pusht", 0.82), ("diffusion_pusht", 0.91)]:
                data = _make_lerobot_eval_info(n_episodes=50, success_rate=sr)
                path = os.path.join(tmpdir, name, "eval_info.json")
                os.makedirs(os.path.dirname(path))
                with open(path, "w") as f:
                    json.dump(data, f)
                files.append(path)

            result = analyze_results(files)
            assert len(result.policies) == 2
            # The second policy should have higher success rate
            assert result.policies[1].success_rates[0] > result.policies[0].success_rates[0]

    def test_significance_test_on_episode_data(self):
        """Run significance test on per-episode binary outcomes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for name, sr in [("bad_policy", 0.3), ("good_policy", 0.9)]:
                data = _make_lerobot_eval_info(n_episodes=50, success_rate=sr)
                path = os.path.join(tmpdir, name, "eval_info.json")
                os.makedirs(os.path.dirname(path))
                with open(path, "w") as f:
                    json.dump(data, f)
                files.append(path)

            result = analyze_results(files)
            ep_a = [float(ep["success"]) for ep in result.policies[0].episodes]
            ep_b = [float(ep["success"]) for ep in result.policies[1].episodes]
            test = significance_test(ep_a, ep_b)
            assert test["significant"] is True
            assert test["p_value"] < 0.001


# ============================================================
# CLI integration tests
# ============================================================

class TestCLI:
    def test_version(self):
        """lerobot-bench --version should work."""
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_bench.cli", "--version"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_help(self):
        """lerobot-bench --help shows all commands."""
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_bench.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "compare" in result.stdout
        assert "analyze" in result.stdout
        assert "sweep" in result.stdout
        assert "regression" in result.stdout

    def test_analyze_help(self):
        """lerobot-bench analyze --help is clear."""
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_bench.cli", "analyze", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "eval_info.json" in result.stdout

    def test_analyze_end_to_end(self):
        """Full CLI test: lerobot-bench analyze with real fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for name, sr in [("policy_a", 0.85), ("policy_b", 0.65)]:
                data = _make_lerobot_eval_info(n_episodes=30, success_rate=sr)
                path = os.path.join(tmpdir, name, "eval_info.json")
                os.makedirs(os.path.dirname(path))
                with open(path, "w") as f:
                    json.dump(data, f)
                files.append(path)

            result = subprocess.run(
                [sys.executable, "-m", "lerobot_bench.cli", "analyze"] + files,
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            # Should produce output (rich table)
            assert "policy_a" in result.stdout or "Policy" in result.stdout

    def test_analyze_markdown_format(self):
        """CLI analyze with --format markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _make_lerobot_eval_info(n_episodes=10, success_rate=0.7)
            path = os.path.join(tmpdir, "my_policy", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(data, f)

            result = subprocess.run(
                [sys.executable, "-m", "lerobot_bench.cli", "analyze", path, "--format", "markdown"],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert "|" in result.stdout  # markdown table

    def test_analyze_nonexistent_file(self):
        """CLI analyze with a file that doesn't exist."""
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_bench.cli", "analyze", "/nonexistent/file.json"],
            capture_output=True, text=True,
        )
        # Should not crash — just produce empty output
        assert result.returncode == 0

    def test_analyze_malformed_json_file(self):
        """CLI analyze with malformed JSON doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad", "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write("{{{{not json")

            result = subprocess.run(
                [sys.executable, "-m", "lerobot_bench.cli", "analyze", path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0

    def test_compare_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lerobot_bench.cli", "compare", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--n-episodes" in result.stdout
        assert "--seeds" in result.stdout
