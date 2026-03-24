"""
analysis/statistics.py
-----------------------
Statistical analysis utilities for comparing FVS algorithm performance.

All functions handle edge cases: empty lists, zero variance, NaN values.
"""

import logging
import math
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pairwise tests
# ---------------------------------------------------------------------------

def wilcoxon_test(scores_a: list, scores_b: list) -> dict:
    """
    Wilcoxon signed-rank test for paired comparisons.

    Args:
        scores_a: Performance scores for algorithm A.
        scores_b: Performance scores for algorithm B (same order as A).

    Returns:
        dict: {statistic, p_value, significant (p < 0.05)}
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False,
                "note": "Insufficient data"}

    diff = a - b
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "note": "All differences are zero"}

    try:
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except Exception as exc:
        logger.warning("Wilcoxon test failed: %s", exc)
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}

    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def friedman_test(*score_groups) -> dict:
    """
    Friedman test for comparing multiple algorithms across multiple instances.

    Args:
        *score_groups: One list per algorithm; each list has one score per instance.

    Returns:
        dict: {statistic, p_value, significant}
    """
    # Convert to arrays and align lengths
    arrays = [np.asarray(g, dtype=float) for g in score_groups]
    min_len = min(len(a) for a in arrays)
    if min_len < 2 or len(arrays) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False,
                "note": "Insufficient data"}

    arrays = [a[:min_len] for a in arrays]
    try:
        stat, p = stats.friedmanchisquare(*arrays)
    except Exception as exc:
        logger.warning("Friedman test failed: %s", exc)
        return {"statistic": float("nan"), "p_value": float("nan"), "significant": False}

    return {"statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


# ---------------------------------------------------------------------------
# Corrections and effect sizes
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: list) -> list:
    """
    Bonferroni correction for multiple comparisons.

    Multiplies each p-value by the number of comparisons and clips to [0, 1].

    Returns:
        List of corrected p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []
    return [min(1.0, p * n) for p in p_values]


def cohens_d(group_a: list, group_b: list) -> float:
    """
    Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns 0.0 if either group has zero variance.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


# ---------------------------------------------------------------------------
# Confidence intervals and summary statistics
# ---------------------------------------------------------------------------

def confidence_interval_95(data: list) -> tuple:
    """
    95% confidence interval using the t-distribution.

    Returns:
        (lower, upper) bounds.  Returns (nan, nan) if data is insufficient.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan"), float("nan")
    mean = np.mean(arr)
    se   = stats.sem(arr)
    ci   = stats.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=se)
    return float(ci[0]), float(ci[1])


def compute_summary_stats(data: list) -> dict:
    """
    Comprehensive summary statistics for a data list.

    Returns:
        dict: mean, median, std, min, max, q25, q75, ci_lower, ci_upper
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        nan = float("nan")
        return dict(mean=nan, median=nan, std=nan, min=nan, max=nan,
                    q25=nan, q75=nan, ci_lower=nan, ci_upper=nan)

    ci_lo, ci_hi = confidence_interval_95(arr.tolist())
    return {
        "mean":     float(np.mean(arr)),
        "median":   float(np.median(arr)),
        "std":      float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":      float(np.min(arr)),
        "max":      float(np.max(arr)),
        "q25":      float(np.percentile(arr, 25)),
        "q75":      float(np.percentile(arr, 75)),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }
