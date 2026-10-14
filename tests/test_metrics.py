import math

import pytest

from ecomrec.metrics import (
    catalogue_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relative_uplift,
    revenue_at_k,
)


def test_precision_at_k_on_toy_case():
    assert precision_at_k([1, 2, 3], {2, 5}, k=3) == pytest.approx(1 / 3)


def test_precision_uses_only_top_k():
    assert precision_at_k([9, 10, 2, 3], {2, 3}, k=2) == 0.0
    assert precision_at_k([2, 3, 7, 8], {2, 3}, k=2) == 1.0


def test_precision_divides_by_k_even_when_list_is_short():
    assert precision_at_k([1], {1}, k=4) == pytest.approx(1 / 4)


def test_recall_at_k_on_toy_case():
    assert recall_at_k([1, 2, 3], {2, 5}, k=3) == pytest.approx(1 / 2)


def test_recall_is_one_when_all_relevant_retrieved():
    assert recall_at_k([5, 2, 8], {2, 5}, k=3) == 1.0


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([2, 5, 9], {2, 5}, k=3) == pytest.approx(1.0)


def test_ndcg_on_toy_case():
    # relevant item at rank 2 only: DCG = 1/log2(3); ideal (one relevant) = 1/log2(2) = 1
    assert ndcg_at_k([1, 2, 3], {2}, k=3) == pytest.approx(1 / math.log2(3))


def test_ndcg_two_relevant_ranks_one_and_three():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([2, 9, 5], {2, 5}, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_zero_when_nothing_relevant_retrieved():
    assert ndcg_at_k([1, 2, 3], {7}, k=3) == 0.0


def test_ndcg_ideal_dcg_is_capped_at_k():
    # 5 relevant items but k=2: a list with 2 hits is perfect.
    assert ndcg_at_k([1, 2], {1, 2, 3, 4, 5}, k=2) == pytest.approx(1.0)


# ---- engagement and revenue proxies --------------------------------------------------


def test_hit_rate_is_one_when_any_recommendation_is_relevant():
    assert hit_rate_at_k([1, 2, 3], {3, 9}, k=3) == 1.0


def test_hit_rate_ignores_hits_beyond_k_and_empty_truth():
    assert hit_rate_at_k([1, 2, 3], {3}, k=2) == 0.0
    assert hit_rate_at_k([1, 2, 3], set(), k=3) == 0.0


def test_revenue_at_k_sums_unit_prices_of_recommended_items_actually_bought():
    prices = {1: 5.0, 2: 2.0, 3: 4.0, 9: 100.0}
    # item 9 was bought but not recommended, item 1 was recommended but not bought
    assert revenue_at_k([1, 2, 3], {2, 3, 9}, prices, k=3) == pytest.approx(6.0)


def test_revenue_at_k_respects_k_and_missing_prices():
    prices = {1: 5.0, 2: 2.0}
    assert revenue_at_k([2, 1, 3], {1, 2, 3}, prices, k=2) == pytest.approx(7.0)
    assert revenue_at_k([3], {3}, prices, k=1) == 0.0  # unknown price counts as zero


def test_catalogue_coverage_is_distinct_recommended_items_over_catalogue_size():
    assert catalogue_coverage([[1, 2], [2, 3]], n_items=10) == pytest.approx(0.3)


def test_catalogue_coverage_needs_a_catalogue():
    with pytest.raises(ValueError):
        catalogue_coverage([[1]], n_items=0)


def test_relative_uplift():
    assert relative_uplift(0.06, 0.04) == pytest.approx(0.5)
    assert relative_uplift(0.03, 0.04) == pytest.approx(-0.25)
    assert math.isnan(relative_uplift(1.0, 0.0))
