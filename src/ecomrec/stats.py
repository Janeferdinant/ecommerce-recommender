"""Small helpers for reporting uncertainty."""
from collections.abc import Sequence

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
