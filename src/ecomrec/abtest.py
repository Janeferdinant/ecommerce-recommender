"""Planning and simulating a randomised experiment (A/B test) for a recommender.

Everything here is a **simulation under stated assumptions**, not a measured business effect:

* the outcome of a customer under each arm comes from an offline replay of historical purchases (what the customer
  bought later, scored against the list each model would have shown);
* the replay treats a later purchase of a recommended product as caused by the list, which a real experiment would
  have to establish;
* customers are drawn independently from the held-out customers (no interference between customers, no novelty
  effect, no drift over time).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _z(alpha: float, power: float) -> tuple[float, float]:
    return float(stats.norm.ppf(1 - alpha / 2)), float(stats.norm.ppf(power))


def required_n_per_arm_proportions(p0: float, relative_uplift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers per arm to detect ``p0 -> p0 * (1 + uplift)`` with a two-sided pooled two-proportion z-test."""
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must be strictly between 0 and 1")
    p1 = p0 * (1.0 + relative_uplift)
    if relative_uplift == 0.0 or not 0.0 < p1 < 1.0:
        raise ValueError("the uplift must be non-zero and keep the rate inside (0, 1)")
    z_a, z_b = _z(alpha, power)
    pbar = (p0 + p1) / 2.0
    numerator = z_a * math.sqrt(2.0 * pbar * (1.0 - pbar)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return math.ceil(numerator**2 / (p1 - p0) ** 2)


def required_n_per_arm_means(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Customers per arm to detect a difference ``delta`` in means (equal standard deviation ``sigma``)."""
    if sigma <= 0 or delta == 0:
        raise ValueError("sigma must be positive and delta non-zero")
    z_a, z_b = _z(alpha, power)
    return math.ceil(2.0 * (z_a + z_b) ** 2 * sigma**2 / delta**2)


def minimum_detectable_relative_uplift(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest relative uplift detectable with ``n_per_arm`` customers per arm (bisection)."""
    lo, hi = 1e-4, min(1.0 / p0 - 1.0, 50.0) * 0.999
    if required_n_per_arm_proportions(p0, hi, alpha, power) > n_per_arm:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if required_n_per_arm_proportions(p0, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi


def _p_values(a: np.ndarray, b: np.ndarray, kind: str) -> np.ndarray:
    """Two-sided p-values for each simulated experiment (rows are experiments)."""
    n = a.shape[1]
    ma, mb = a.mean(axis=1), b.mean(axis=1)
    if kind == "proportion":
        pooled = (ma + mb) / 2.0
        se = np.sqrt(np.maximum(pooled * (1 - pooled) * 2.0 / n, 1e-300))
    elif kind == "mean":
        se = np.sqrt(np.maximum(a.var(axis=1, ddof=1) / n + b.var(axis=1, ddof=1) / n, 1e-300))
    else:
        raise ValueError("kind must be 'proportion' or 'mean'")
    z = (mb - ma) / se
    return np.asarray(2.0 * stats.norm.sf(np.abs(z)))


def simulate_power(control: np.ndarray, treatment: np.ndarray, n_per_arm: int, n_sims: int = 2000, seed: int = 0,
                   alpha: float = 0.05, kind: str = "proportion") -> float:
    """Share of simulated experiments that reject the null.

    ``control[i]`` and ``treatment[i]`` are customer ``i``'s outcome under each arm (replay). Each experiment draws
    ``n_per_arm`` customers for each arm independently with replacement and compares the arms with a z-test
    (``proportion``) or a Welch test with a normal reference (``mean``).
    """
    c, t = np.asarray(control, dtype=np.float64), np.asarray(treatment, dtype=np.float64)
    if len(c) != len(t) or len(c) == 0:
        raise ValueError("control and treatment need the same, non-zero number of customers")
    rng = np.random.default_rng(seed)
    rejected = 0
    chunk = max(1, min(n_sims, 4_000_000 // max(n_per_arm, 1)))
    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        a = c[rng.integers(0, len(c), size=(m, n_per_arm))]
        b = t[rng.integers(0, len(t), size=(m, n_per_arm))]
        rejected += int((_p_values(a, b, kind) < alpha).sum())
        done += m
    return rejected / n_sims


def aa_false_positive_rate(outcome: np.ndarray, n_per_arm: int, n_sims: int = 2000, seed: int = 0, alpha: float = 0.05,
                           kind: str = "proportion") -> float:
    """Rejection rate when both arms see the same policy; it should be close to ``alpha``."""
    return simulate_power(outcome, outcome, n_per_arm, n_sims, seed, alpha, kind)
