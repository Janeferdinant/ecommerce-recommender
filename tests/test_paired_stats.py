"""Paired bootstrap comparison of two models and a bootstrap interval for a mean."""
import numpy as np
import pytest

from ecomrec.stats import bootstrap_mean_ci, paired_bootstrap_difference


def test_identical_models_have_zero_difference_and_p_one():
    a = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0] * 20)
    out = paired_bootstrap_difference(a, a.copy(), n_boot=500, seed=0)
    assert out["diff"] == 0.0 and out["diff_ci"] == (0.0, 0.0) and out["p_two_sided"] == 1.0


def test_a_clearly_better_model_has_an_interval_above_zero():
    rng = np.random.default_rng(0)
    base = rng.binomial(1, 0.3, 800).astype(float)
    better = np.clip(base + rng.binomial(1, 0.15, 800), 0, 1)
    out = paired_bootstrap_difference(better, base, n_boot=1000, seed=1)
    assert out["diff"] > 0 and out["diff_ci"][0] > 0 and out["rel_diff_ci"][0] > 0
    assert out["p_two_sided"] < 0.01


def test_noise_only_difference_spans_zero():
    rng = np.random.default_rng(2)
    a = rng.binomial(1, 0.3, 400).astype(float)
    b = rng.binomial(1, 0.3, 400).astype(float)
    out = paired_bootstrap_difference(a, b, n_boot=1000, seed=3)
    assert out["diff_ci"][0] < 0 < out["diff_ci"][1]


def test_difference_ci_is_narrower_when_the_pairs_are_correlated():
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 1.0, 500)
    paired = paired_bootstrap_difference(base + 0.1 + rng.normal(0, 0.05, 500), base, n_boot=500, seed=5)
    unpaired = paired_bootstrap_difference(rng.normal(0.1, 1.0, 500), base, n_boot=500, seed=5)
    assert paired["diff_ci"][1] - paired["diff_ci"][0] < unpaired["diff_ci"][1] - unpaired["diff_ci"][0]


def test_paired_bootstrap_is_deterministic_and_validates_input():
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    assert paired_bootstrap_difference(a, b, n_boot=100, seed=7) == paired_bootstrap_difference(a, b, n_boot=100, seed=7)
    with pytest.raises(ValueError):
        paired_bootstrap_difference(a, b[:5])


def test_mean_ci_brackets_the_mean():
    values = np.random.default_rng(0).normal(5.0, 1.0, 300)
    lo, hi = bootstrap_mean_ci(values, n_boot=500, seed=0)
    assert lo < values.mean() < hi and hi - lo < 0.5
