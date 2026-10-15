"""Small helpers for reporting uncertainty."""
from collections.abc import Sequence
from typing import Any

import numpy as np


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; zero for a single value)."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def bootstrap_uplift_ci(model_values: Sequence[float], baseline_values: Sequence[float],
                        n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Paired bootstrap over customers for ``mean(model) / mean(baseline) - 1``.

    The same resampled customers are used for the model and the baseline, so the
    interval reflects customer-sampling uncertainty of the *relative* uplift.
    """
    model = np.asarray(model_values, dtype=np.float64)
    base = np.asarray(baseline_values, dtype=np.float64)
    if len(model) != len(base) or len(model) == 0:
        raise ValueError("model and baseline need the same, non-zero number of customers")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(model), size=(n_boot, len(model)))
    ratio = model[idx].mean(axis=1) / base[idx].mean(axis=1) - 1.0
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(ratio, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for a mean over customers."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("need at least one value")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_bootstrap_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0,
                                level: float = 0.95) -> dict[str, Any]:
    """Paired bootstrap over customers for ``mean(a) - mean(b)``.

    Both models are scored on the same resampled customers. Returns the absolute difference with its interval,
    the relative difference ``mean(a) / mean(b) - 1`` with its interval and a two-sided bootstrap p-value
    (twice the smaller tail of resampled differences on either side of zero, at least ``1 / n_boot``).
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("a and b need the same, non-zero number of customers")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    mean_a, mean_b = x[idx].mean(axis=1), y[idx].mean(axis=1)
    diff = mean_a - mean_b
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = mean_a / mean_b - 1.0
    alpha = (1.0 - level) / 2.0
    d_lo, d_hi = np.quantile(diff, [alpha, 1.0 - alpha])
    r_lo, r_hi = np.nanquantile(rel, [alpha, 1.0 - alpha])
    p = 2.0 * min(float(np.mean(diff <= 0.0)), float(np.mean(diff >= 0.0)))
    base = float(y.mean())
    return {
        "mean_a": float(x.mean()),
        "mean_b": base,
        "diff": float(x.mean() - y.mean()),
        "diff_ci": (float(d_lo), float(d_hi)),
        "rel_diff": float(x.mean() / base - 1.0) if base else float("nan"),
        "rel_diff_ci": (float(r_lo), float(r_hi)),
        "p_two_sided": float(min(1.0, max(p, 1.0 / n_boot))) if p < 1.0 else 1.0,
    }
