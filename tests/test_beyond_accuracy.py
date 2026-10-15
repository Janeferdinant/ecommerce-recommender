"""Novelty, diversity, popularity-bias and revenue-calibration metrics on toy cases."""
import math

import numpy as np
import pytest

from ecomrec.metrics import (
    assign_popularity_bins,
    intra_list_diversity,
    jaccard_distance,
    novelty_at_k,
    oracle_revenue_at_k,
    popularity_bin_recall,
    popularity_bin_shares,
)


def test_novelty_is_mean_self_information_in_bits():
    buyers = {"a": 50, "b": 25, "c": 100}
    # p = 0.5 -> 1 bit, p = 0.25 -> 2 bits
    assert novelty_at_k(["a", "b"], buyers, n_users=100, k=2) == pytest.approx(1.5)
    assert novelty_at_k(["c"], buyers, n_users=100, k=1) == pytest.approx(0.0)


def test_novelty_only_looks_at_the_top_k():
    buyers = {"a": 50, "b": 25}
    assert novelty_at_k(["a", "b"], buyers, n_users=100, k=1) == pytest.approx(1.0)


def test_novelty_of_an_empty_list_is_zero():
    assert novelty_at_k([], {"a": 1}, n_users=10, k=5) == 0.0


def test_diversity_is_one_minus_mean_pairwise_similarity():
    assert intra_list_diversity(["a", "b", "c"], lambda x, y: 1.0, k=3) == pytest.approx(0.0)
    assert intra_list_diversity(["a", "b", "c"], lambda x, y: 0.0, k=3) == pytest.approx(1.0)
    sims = {frozenset("ab"): 0.8, frozenset("ac"): 0.2, frozenset("bc"): 0.5}
    assert intra_list_diversity(["a", "b", "c"], lambda x, y: sims[frozenset((x, y))], k=3) == pytest.approx(0.5)


def test_diversity_needs_two_items():
    assert intra_list_diversity(["a"], lambda x, y: 1.0, k=5) == 0.0


def test_jaccard_distance():
    assert jaccard_distance(frozenset("ab"), frozenset("bc")) == pytest.approx(1 - 1 / 3)
    assert jaccard_distance(frozenset("ab"), frozenset("ab")) == 0.0
    assert jaccard_distance(frozenset(), frozenset()) == 0.0


def test_popularity_bins_put_best_sellers_first_and_are_equal_sized():
    buyers = {f"i{n}": 100 - n for n in range(10)}
    bins = assign_popularity_bins(buyers, n_bins=5)
    assert bins["i0"] == 0 and bins["i1"] == 0 and bins["i9"] == 4
    assert [sum(1 for b in bins.values() if b == k) for k in range(5)] == [2, 2, 2, 2, 2]


def test_popularity_bins_are_deterministic_under_ties():
    buyers = {"b": 5, "a": 5, "c": 1}
    assert assign_popularity_bins(buyers, n_bins=3) == {"a": 0, "b": 1, "c": 2}


def test_bin_shares_sum_to_one():
    bins = {"a": 0, "b": 0, "c": 1, "d": 1}
    shares = popularity_bin_shares([["a", "b"], ["a", "c"]], bins, n_bins=2, k=2)
    assert shares.tolist() == [0.75, 0.25]


def test_bin_recall_counts_relevant_items_found_per_bin():
    bins = {"a": 0, "b": 0, "c": 1, "d": 1}
    recs = [["a", "c"], ["b", "x"]]
    truth = [{"a", "d"}, {"b", "c"}]
    recall = popularity_bin_recall(recs, truth, bins, n_bins=2, k=2)
    # bin 0: relevant a, b, both found; bin 1: relevant d (missed), c (missed for user 2, found for user 1? no: c not relevant to 1)
    assert recall[0] == pytest.approx(1.0)
    assert recall[1] == pytest.approx(0.0)


def test_bin_recall_is_nan_for_empty_bins():
    recall = popularity_bin_recall([["a"]], [{"a"}], {"a": 0, "b": 1}, n_bins=2, k=1)
    assert recall[0] == 1.0 and math.isnan(recall[1])


def test_relevant_items_outside_the_catalogue_are_ignored_by_bin_recall():
    recall = popularity_bin_recall([["a"]], [{"a", "new"}], {"a": 0}, n_bins=1, k=1)
    assert recall[0] == 1.0


def test_oracle_revenue_is_the_best_possible_value_in_k_slots():
    prices = {"a": 10.0, "b": 3.0, "c": 7.0}
    assert oracle_revenue_at_k({"a", "b", "c"}, prices, k=2) == pytest.approx(17.0)
    assert oracle_revenue_at_k({"b"}, prices, k=2) == pytest.approx(3.0)
    assert oracle_revenue_at_k(set(), prices, k=2) == 0.0
    assert np.isfinite(oracle_revenue_at_k({"zzz"}, prices, k=1))  # unknown price counts as zero
