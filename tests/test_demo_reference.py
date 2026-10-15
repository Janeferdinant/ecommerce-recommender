"""Reference implementation of the demo's picker mode (the page's JavaScript must reproduce it)."""
import numpy as np
import pytest

from ecomrec.demo_ref import recommend_from_exported, top_neighbours


def tiny_data(with_price=True):
    lists = [
        [[1, 0.9], [2, 0.5]],  # source 0 points to items 1 and 2
        [[0, 0.9], [3, 0.4]],
        [[0, 0.5], [3, 0.8]],
        [[1, 0.4], [2, 0.8]],
        [],
    ]
    data = {
        "catalogue": [0, 1, 2, 3, 4],
        "lists": {"itemknn": lists, "itemknn_recency": lists, "ease": lists, "copurchase": lists},
        "als": {"alpha": 10.0, "reg": 1.0, "factors": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]],
                "gram": [[2.0, 0.0], [0.0, 2.0]]},
        "bpr": {"factors": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]], "bias": [0.0, 0.1, 0.0, 0.0, 0.0]},
    }
    if with_price:
        data["price_weight"] = [1.0, 1.0, 4.0, 0.25, 1.0]
    return data


def ids(recs):
    return [r["i"] for r in recs]


def test_popularity_follows_catalogue_order_and_skips_the_basket():
    out = recommend_from_exported(tiny_data(), [0], k=3)
    assert ids(out["popularity"]) == [1, 2, 3]
    assert out["popularity"][0]["w"] == []


def test_neighbour_models_sum_scores_over_basket_items():
    out = recommend_from_exported(tiny_data(), [0, 3], k=4)
    # item 1: 0.9 (from 0) + 0.4 (from 3) = 1.3; item 2: 0.5 + 0.8 = 1.3 ties with item 1 -> lower index first
    assert ids(out["itemknn"])[:2] == [1, 2]
    top = out["itemknn"][0]["w"]
    assert [src for src, _ in top] == [0, 3] and sum(share for _, share in top) == pytest.approx(1.0, abs=0.02)


def test_backfill_uses_popular_items_when_neighbours_run_out():
    out = recommend_from_exported(tiny_data(), [4], k=3)  # item 4 has no neighbours
    assert ids(out["itemknn"]) == [0, 1, 2]
    assert out["itemknn"][0]["w"] == []


def test_price_model_reorders_by_the_price_weight():
    out = recommend_from_exported(tiny_data(), [0, 3], k=2)
    assert ids(out["copurchase"])[:2] == [1, 2]
    assert ids(out["copurchase_price"])[0] == 2  # 1.3 * 4.0 beats 1.3 * 1.0
    assert out["copurchase_price"][0]["f"] == pytest.approx(4.0)


def test_no_price_model_without_price_weights():
    assert "copurchase_price" not in recommend_from_exported(tiny_data(with_price=False), [0], k=2)


def test_factor_models_return_k_new_items_with_contributors():
    out = recommend_from_exported(tiny_data(), [0], k=3)
    for model in ("als", "bpr"):
        recs = out[model]
        assert len(recs) == 3 and 0 not in ids(recs) and len(set(ids(recs))) == 3
        assert all(src == 0 for r in recs for src, _ in r["w"])
    assert ids(out["bpr"])[0] == 1  # closest factor vector plus the bias


def test_results_do_not_depend_on_basket_order():
    a = recommend_from_exported(tiny_data(), [0, 3], k=3)
    b = recommend_from_exported(tiny_data(), [3, 0], k=3)
    assert a == b


def test_top_neighbours_keeps_the_largest_positive_entries():
    matrix = np.array([[0.0, 0.5, 0.2, -0.3], [0.5, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.0, 0.0]])
    rows = top_neighbours(matrix, n=2)
    assert rows[0] == [[1, 0.5], [2, 0.2]]
    assert rows[1] == [[0, 0.5], [3, 0.1]]
    assert rows[2] == []
    assert rows[3] == [[0, 0.2], [1, 0.2]]  # ties: lower index first
