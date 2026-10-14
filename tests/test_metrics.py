import math

import pytest

from ecomrec.metrics import ndcg_at_k, precision_at_k, recall_at_k


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
