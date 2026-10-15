"""The extended evaluation must reproduce the earlier harness and add beyond-accuracy measures."""
import numpy as np
import pytest

from ecomrec.evaluate import evaluate_detailed
from ecomrec.evaluation import evaluate_extended
from ecomrec.models import ALSModel, ItemKNNModel, PopularityModel
from ecomrec.neighbours import CoPurchaseModel

K = 5


def models():
    return [PopularityModel(), ItemKNNModel(), ALSModel(factors=6, iterations=4, seed=3)]


def test_it_reproduces_the_earlier_harness_exactly(split):
    old, old_per_user = evaluate_detailed(split, seed=3, k=K, models=models())
    new = evaluate_extended(split, models(), k=K, seed=3)
    assert new.users == sorted(new.users) and len(new.users) == old["n_eval_users"]
    for name, metrics in old["models"].items():
        for metric, value in metrics.items():
            assert new.summary[name][metric] == pytest.approx(value, abs=1e-12), (name, metric)
        for metric, values in old_per_user[name].items():
            np.testing.assert_allclose(new.per_user[name][metric], values)


def test_beyond_accuracy_metrics_are_present_and_in_range(split):
    new = evaluate_extended(split, models(), k=K, seed=3)
    for name in ("popularity", "itemknn", "als"):
        s = new.summary[name]
        assert s[f"novelty@{K}"] > 0
        assert 0.0 <= s[f"diversity@{K}"] <= 1.0
        assert 0.0 <= s[f"revenue_capture@{K}"] <= 1.0
        assert len(new.per_user[name][f"diversity@{K}"]) == len(new.users)
    assert new.summary["popularity"][f"novelty@{K}"] < new.summary["itemknn"][f"novelty@{K}"]


def test_popularity_curves_have_one_value_per_bin(split):
    new = evaluate_extended(split, models(), k=K, seed=3, n_bins=5)
    for name in ("popularity", "als"):
        curves = new.popularity_curves[name]
        assert len(curves["rec_share"]) == len(curves["recall"]) == 5
        assert sum(curves["rec_share"]) == pytest.approx(1.0)
    shares = new.popularity_curves["popularity"]["rec_share"]
    assert shares[0] == max(shares)  # the baseline concentrates on the best-selling bin


def test_revenue_sensitivity_uses_test_prices_but_only_as_a_check(split):
    new = evaluate_extended(split, models(), k=K, seed=3)
    assert f"revenue_testprice@{K}" in new.summary["popularity"]
    assert "price_drift_median" in new.meta and new.meta["price_drift_median"] > 0


def test_genre_diversity_is_reported_when_attributes_are_given(split):
    train, _ = split
    genres = {item: frozenset({f"g{int(item[1:]) % 4}"}) for item in train["item"].unique()}
    new = evaluate_extended(split, models(), k=K, seed=3, genres=genres)
    assert 0.0 <= new.summary["als"][f"genre_diversity@{K}"] <= 1.0


def test_beyond_false_skips_the_expensive_parts(split):
    fast = evaluate_extended(split, models(), k=K, seed=3, beyond=False)
    assert f"diversity@{K}" not in fast.summary["als"] and fast.popularity_curves == {}
    full = evaluate_extended(split, models(), k=K, seed=3)
    assert fast.summary["als"][f"ndcg@{K}"] == pytest.approx(full.summary["als"][f"ndcg@{K}"])


def test_price_free_data_skips_revenue_columns(split):
    train, test = split
    no_price = (train.drop(columns=["price"]), test.drop(columns=["price"]))
    new = evaluate_extended(no_price, [PopularityModel()], k=K, seed=0)
    assert f"revenue@{K}" not in new.summary["popularity"]


def test_works_with_the_copurchase_model(split):
    new = evaluate_extended(split, [CoPurchaseModel(min_support=1, gamma=0.5)], k=K, seed=0)
    assert "copurchase_price" in new.summary
