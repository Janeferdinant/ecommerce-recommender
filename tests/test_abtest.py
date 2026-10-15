"""Sample-size formulas and the randomised-experiment simulation (a simulation, never a measured effect)."""
import numpy as np
import pytest

from ecomrec.abtest import (
    aa_false_positive_rate,
    minimum_detectable_relative_uplift,
    required_n_per_arm_means,
    required_n_per_arm_proportions,
    simulate_power,
)


def test_sample_size_for_two_proportions_matches_the_textbook_value():
    # 10% -> 12% with alpha 0.05 (two-sided) and 80% power needs about 3,841 customers per arm
    assert abs(required_n_per_arm_proportions(0.10, 0.20) - 3841) <= 2


def test_sample_size_for_means_matches_the_textbook_value():
    # detecting 0.1 sd with alpha 0.05 and 80% power needs about 1,570 per arm
    assert abs(required_n_per_arm_means(sigma=1.0, delta=0.1) - 1570) <= 2


def test_smaller_effects_and_higher_power_need_more_customers():
    base = required_n_per_arm_proportions(0.05, 0.10)
    assert required_n_per_arm_proportions(0.05, 0.05) > base
    assert required_n_per_arm_proportions(0.05, 0.10, power=0.9) > base
    assert required_n_per_arm_proportions(0.05, 0.10, alpha=0.01) > base


def test_invalid_inputs_are_rejected():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            required_n_per_arm_proportions(bad, 0.1)
    with pytest.raises(ValueError):
        required_n_per_arm_proportions(0.1, 0.0)
    with pytest.raises(ValueError):
        required_n_per_arm_means(sigma=0.0, delta=1.0)


def test_minimum_detectable_uplift_inverts_the_sample_size():
    n = required_n_per_arm_proportions(0.08, 0.15)
    assert minimum_detectable_relative_uplift(0.08, n) == pytest.approx(0.15, abs=0.005)


def test_simulated_power_agrees_with_the_formula():
    rng = np.random.default_rng(0)
    control = rng.binomial(1, 0.20, 20000).astype(float)
    treatment = control.copy()
    flip = rng.random(20000) < 0.05  # treatment converts an extra ~5 points
    treatment[flip] = 1.0
    p0, p1 = control.mean(), treatment.mean()
    n = required_n_per_arm_proportions(p0, p1 / p0 - 1.0, power=0.8)
    power = simulate_power(control, treatment, n_per_arm=n, n_sims=800, seed=1, kind="proportion")
    assert 0.72 <= power <= 0.88


def test_power_grows_with_sample_size():
    rng = np.random.default_rng(1)
    control = rng.binomial(1, 0.2, 5000).astype(float)
    treatment = np.clip(control + (rng.random(5000) < 0.04), 0, 1)
    small = simulate_power(control, treatment, 200, n_sims=300, seed=2)
    large = simulate_power(control, treatment, 3000, n_sims=300, seed=2)
    assert large > small


def test_continuous_outcomes_use_a_welch_test():
    rng = np.random.default_rng(3)
    control = rng.exponential(2.0, 10000)
    treatment = control * 1.15
    assert simulate_power(control, treatment, 2500, n_sims=300, seed=1, kind="mean") > 0.5


def test_a_a_test_rejects_about_alpha_of_the_time():
    rng = np.random.default_rng(5)
    outcome = rng.binomial(1, 0.3, 8000).astype(float)
    rate = aa_false_positive_rate(outcome, n_per_arm=1000, n_sims=1500, seed=2)
    assert 0.02 <= rate <= 0.08


def test_simulation_is_deterministic():
    y0 = np.array([0.0, 1.0] * 500)
    y1 = np.array([1.0, 1.0] * 500)
    assert simulate_power(y0, y1, 50, 100, seed=3) == simulate_power(y0, y1, 50, 100, seed=3)
