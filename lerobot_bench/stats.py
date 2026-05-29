"""Statistical analysis for policy comparison."""

import numpy as np
from scipy import stats as sp_stats


def compute_statistics(values, confidence=0.95):
    """Compute mean, std, confidence interval via bootstrap."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    # Bootstrap confidence interval
    n_bootstrap = 10000
    if len(arr) >= 3:
        rng = np.random.default_rng(42)
        boot_means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        alpha = (1 - confidence) / 2
        ci_low = float(np.percentile(boot_means, alpha * 100))
        ci_high = float(np.percentile(boot_means, (1 - alpha) * 100))
    else:
        ci_low = mean - 1.96 * std / max(np.sqrt(len(arr)), 1)
        ci_high = mean + 1.96 * std / max(np.sqrt(len(arr)), 1)

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(arr),
    }


def significance_test(values_a, values_b, method="mann-whitney"):
    """Test if two sets of results are significantly different.

    Returns:
        dict with p_value, significant (bool), effect_size, method
    """
    a = np.array(values_a, dtype=np.float64)
    b = np.array(values_b, dtype=np.float64)

    if len(a) < 3 or len(b) < 3:
        return {
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0.0,
            "method": method,
            "note": "insufficient samples",
        }

    if method == "mann-whitney":
        stat, p_value = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    elif method == "t-test":
        stat, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
    elif method == "wilcoxon":
        # Paired test — requires same length
        min_len = min(len(a), len(b))
        stat, p_value = sp_stats.wilcoxon(a[:min_len], b[:min_len])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cohen's d effect size
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    effect_size = (a.mean() - b.mean()) / pooled_std if pooled_std > 1e-10 else 0.0

    return {
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "effect_size": float(effect_size),
        "method": method,
    }


def detect_regression(comparison_result, threshold=0.05):
    """Check if the second policy regresses vs the first.

    Returns True if candidate (index 1) is more than `threshold` worse than baseline (index 0).
    """
    policies = comparison_result.policies
    if len(policies) < 2:
        return False

    baseline = policies[0]
    candidate = policies[1]

    if not baseline.success_rates or not candidate.success_rates:
        return False

    base_mean = np.mean(baseline.success_rates)
    cand_mean = np.mean(candidate.success_rates)

    if base_mean == 0:
        return False

    regression_amount = (base_mean - cand_mean) / base_mean
    return bool(regression_amount > threshold)


def rank_policies(comparison_result, metric="success_rate"):
    """Rank policies by a metric, return sorted indices and scores."""
    scores = []
    for policy in comparison_result.policies:
        if metric == "success_rate":
            scores.append(np.mean(policy.success_rates) if policy.success_rates else 0.0)
        elif metric == "reward":
            scores.append(np.mean(policy.rewards) if policy.rewards else 0.0)
        else:
            scores.append(0.0)

    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices.tolist(), [scores[i] for i in ranked_indices]
