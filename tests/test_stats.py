"""Tests for statistical analysis."""

import numpy as np
import pytest

from lerobot_bench.stats import compute_statistics, significance_test, detect_regression
from lerobot_bench.runner import PolicyResult, ComparisonResult


def test_compute_statistics_basic():
    values = [0.8, 0.9, 0.85, 0.7, 0.95]
    stats = compute_statistics(values)
    assert abs(stats["mean"] - 0.84) < 0.01
    assert stats["n"] == 5
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]


def test_compute_statistics_empty():
    stats = compute_statistics([])
    assert stats["mean"] == 0.0
    assert stats["n"] == 0


def test_compute_statistics_single():
    stats = compute_statistics([0.5])
    assert stats["mean"] == 0.5
    assert stats["n"] == 1


def test_significance_clearly_different():
    a = [1.0] * 50  # always succeeds
    b = [0.0] * 50  # always fails
    result = significance_test(a, b)
    assert result["significant"] is True
    assert result["p_value"] < 0.001


def test_significance_same_distribution():
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.1, 100).tolist()
    b = rng.normal(0.5, 0.1, 100).tolist()
    result = significance_test(a, b)
    # Should usually NOT be significant (same distribution)
    assert result["p_value"] > 0.01


def test_significance_insufficient_samples():
    result = significance_test([1.0, 0.0], [0.5])
    assert result["significant"] is False
    assert "insufficient" in result.get("note", "")


def test_detect_regression_true():
    baseline = PolicyResult(name="v1", path="", success_rates=[0.9, 0.85, 0.88])
    candidate = PolicyResult(name="v2", path="", success_rates=[0.5, 0.45, 0.48])
    cr = ComparisonResult(policies=[baseline, candidate])
    assert detect_regression(cr, threshold=0.05) is True


def test_detect_regression_false():
    baseline = PolicyResult(name="v1", path="", success_rates=[0.8, 0.82, 0.79])
    candidate = PolicyResult(name="v2", path="", success_rates=[0.85, 0.83, 0.87])
    cr = ComparisonResult(policies=[baseline, candidate])
    assert detect_regression(cr, threshold=0.05) is False
