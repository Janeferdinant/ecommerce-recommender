"""Offline engagement and revenue proxies computed by the evaluation harness."""
import numpy as np
import pandas as pd
import pytest

from ecomrec.evaluate import evaluate, evaluate_detailed
from ecomrec.stats import bootstrap_uplift_ci, mean_sd

T = pd.Timestamp


class Stub:
    """Recommends a fixed list to everyone, so the expected metrics can be done by hand."""

    def __init__(self, name, recs):
        self.name, self._recs = name, recs

    def fit(self, train):
        return self

    def recommend(self, user, k=10):
        return self._recs[:k]


def toy_split():
    train = pd.DataFrame(
        [
            ("a", "x", T("2011-01-01"), 1.0), ("a", "y", T("2011-01-01"), 2.0),
            ("b", "x", T("2011-01-02"), 1.0), ("b", "y", T("2011-01-02"), 2.0),
            ("c", "z", T("2011-01-03"), 10.0), ("c", "w", T("2011-01-03"), 20.0),
        ],
        columns=["user", "item", "ts", "price"],
    )
    test = pd.DataFrame(
        [
            ("a", "z", T("2011-02-01"), 11.0),   # new item for a: relevant
            ("a", "x", T("2011-02-01"), 1.0),    # repeat purchase: not relevant
            ("b", "w", T("2011-02-02"), 25.0),   # new item for b: relevant
            ("d", "z", T("2011-02-03"), 10.0),   # customer unseen in training: not evaluated
        ],
        columns=["user", "item", "ts", "price"],
    )
    return train, test


def test_proxies_on_a_hand_computed_toy_case():
    out = evaluate(toy_split(), k=2, models=[Stub("hits", ["z", "w"]), Stub("misses", ["x", "y"])])
    hits, misses = out["models"]["hits"], out["models"]["misses"]
    assert out["n_eval_users"] == 2
    assert hits["hit_rate@2"] == 1.0 and misses["hit_rate@2"] == 0.0
    # prices come from the TRAIN period (median): z=10, w=20 -> a gets 10, b gets 20
    assert hits["revenue@2"] == pytest.approx(15.0)
    assert misses["revenue@2"] == 0.0
    # 2 distinct recommended items out of 4 training products
    assert hits["coverage@2"] == pytest.approx(0.5)
    assert hits["precision@2"] == pytest.approx(0.5)


def test_prices_for_revenue_never_come_from_the_test_period():
    train, test = toy_split()
    test = test.assign(price=999.0)
    out = evaluate((train, test), k=2, models=[Stub("hits", ["z", "w"])])
    assert out["models"]["hits"]["revenue@2"] == pytest.approx(15.0)


def test_revenue_is_omitted_without_a_price_column():
    train, test = toy_split()
    out = evaluate((train.drop(columns="price"), test.drop(columns="price")), k=2,
                   models=[Stub("hits", ["z", "w"])])
    assert "revenue@2" not in out["models"]["hits"]


def test_evaluate_reports_all_proxy_metrics_for_every_model(split):
    out = evaluate(split, seed=1, als_params={"factors": 8, "iterations": 3})
    for model in ("popularity", "itemknn", "als"):
        assert {"hit_rate@10", "coverage@10", "revenue@10"} <= set(out["models"][model])
        assert 0.0 <= out["models"][model]["coverage@10"] <= 1.0


def test_detailed_evaluation_exposes_per_customer_values(split):
    summary, per_user = evaluate_detailed(split, seed=1, als_params={"factors": 8, "iterations": 3})
    n = summary["n_eval_users"]
    assert len(per_user["als"]["ndcg@10"]) == n
    assert np.mean(per_user["als"]["ndcg@10"]) == pytest.approx(summary["models"]["als"]["ndcg@10"])


def test_mean_sd_uses_sample_standard_deviation():
    mean, sd = mean_sd([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0) and sd == pytest.approx(1.0)
    assert mean_sd([5.0]) == (5.0, 0.0)


def test_bootstrap_uplift_ci_is_deterministic_and_ordered():
    rng = np.random.default_rng(0)
    base = rng.random(200)
    model = base + rng.random(200) * 0.2
    a = bootstrap_uplift_ci(model, base, n_boot=300, seed=3)
    b = bootstrap_uplift_ci(model, base, n_boot=300, seed=3)
    assert a == b
    assert a[0] <= a[1]


def test_bootstrap_uplift_ci_collapses_for_a_constant_ratio():
    base = np.array([0.1, 0.2, 0.3, 0.4])
    lo, hi = bootstrap_uplift_ci(2 * base, base, n_boot=200, seed=0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
