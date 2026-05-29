"""Tests for eval_info.json analyzer."""

import json
import os
import tempfile

import pytest

from lerobot_bench.analyzer import analyze_results


def _make_eval_info(success_rate=0.8, n_episodes=10):
    return {
        "aggregated": {
            "pc_success": success_rate,
            "avg_sum_reward": success_rate * 2.0,
            "eval_s": 30.0,
        },
        "per_episode": [
            {"success": i < int(success_rate * n_episodes), "sum_reward": float(i < int(success_rate * n_episodes)), "steps": 100}
            for i in range(n_episodes)
        ],
    }


def test_analyze_single_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "policy_a", "eval_info.json")
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(_make_eval_info(0.8, 10), f)

        result = analyze_results([path])
        assert len(result.policies) == 1
        assert abs(result.policies[0].success_rates[0] - 0.8) < 0.01


def test_analyze_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        files = []
        for name, sr in [("policy_a", 0.9), ("policy_b", 0.6)]:
            path = os.path.join(tmpdir, name, "eval_info.json")
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                json.dump(_make_eval_info(sr, 20), f)
            files.append(path)

        result = analyze_results(files)
        assert len(result.policies) == 2
        assert result.policies[0].success_rates[0] > result.policies[1].success_rates[0]


def test_analyze_missing_file():
    result = analyze_results(["/nonexistent/path.json"])
    assert len(result.policies) == 0
