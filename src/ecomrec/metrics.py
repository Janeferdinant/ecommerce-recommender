"""Top-k ranking metrics with binary relevance, plus offline engagement and revenue proxies.

The proxies (hit rate, catalogue coverage, revenue) are computed on historical
purchases. They are offline stand-ins, not measurements of deployed engagement
or sales revenue.
"""
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _hits(recommended: Sequence[Any], relevant: set[Any], k: int) -> int:
    return sum(1 for item in list(recommended)[:k] if item in relevant)


def precision_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Share of the top-k slots that hold a relevant item (always divided by k)."""
    return _hits(recommended, set(relevant), k) / k


def recall_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Share of the relevant items found in the top k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return _hits(recommended, relevant, k) / len(relevant)


def ndcg_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(list(recommended)[:k])
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal


def hit_rate_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """1.0 if at least one of the top k is relevant, else 0.0 (engagement proxy)."""
    return 1.0 if _hits(recommended, set(relevant), k) > 0 else 0.0


def revenue_at_k(recommended: Sequence[Any], relevant: Iterable[Any],
                 prices: Mapping[Any, float], k: int) -> float:
    """Unit-price-weighted value of the recommended items the customer really bought.

    Items without a known price count as zero (revenue proxy).
    """
    relevant = set(relevant)
    return float(sum(prices.get(item, 0.0) for item in list(recommended)[:k] if item in relevant))


def catalogue_coverage(recommendation_lists: Iterable[Sequence[Any]], n_items: int) -> float:
    """Distinct items recommended to anyone, as a share of the catalogue (engagement proxy)."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    distinct = {item for recs in recommendation_lists for item in recs}
    return len(distinct) / n_items


def relative_uplift(value: float, baseline: float) -> float:
    """``value / baseline - 1``; NaN when the baseline is zero."""
    if baseline == 0:
        return math.nan
    return value / baseline - 1.0


# ---- beyond accuracy ---------------------------------------------------------
def novelty_at_k(recommended: Sequence[Any], buyers: Mapping[Any, int], n_users: int, k: int) -> float:
    """Mean self-information (bits) of the top-k items: ``-log2(share of customers who bought the item)``.

    Higher means the list leans towards less popular items. Empty lists score 0.
    """
    items = [i for i in list(recommended)[:k] if buyers.get(i, 0) > 0]
    if not items:
        return 0.0
    return sum(-math.log2(buyers[i] / n_users) for i in items) / len(items)


def jaccard_distance(a: frozenset[Any], b: frozenset[Any]) -> float:
    """``1 - |a & b| / |a | b|`` (0 when both sets are empty)."""
    union = a | b
    return 1.0 - len(a & b) / len(union) if union else 0.0


def intra_list_diversity(recommended: Sequence[Any], similarity: Callable[[Any, Any], float], k: int) -> float:
    """``1 - mean pairwise similarity`` over the top-k items (0 for fewer than two items)."""
    items = list(recommended)[:k]
    if len(items) < 2:
        return 0.0
    pairs = [(items[a], items[b]) for a in range(len(items)) for b in range(a + 1, len(items))]
    return 1.0 - sum(similarity(x, y) for x, y in pairs) / len(pairs)


def assign_popularity_bins(buyers: Mapping[Any, int], n_bins: int = 10) -> dict[Any, int]:
    """Equal-sized bins of the catalogue by popularity; bin 0 holds the best sellers (ties: item order)."""
    ranked = sorted(buyers, key=lambda item: (-buyers[item], str(item)))
    n = len(ranked)
    return {item: min(rank * n_bins // n, n_bins - 1) for rank, item in enumerate(ranked)}


def popularity_bin_shares(recommendation_lists: Iterable[Sequence[Any]], bins: Mapping[Any, int], n_bins: int,
                          k: int) -> np.ndarray:
    """Share of all recommended slots that fall into each popularity bin."""
    counts = np.zeros(n_bins)
    for recs in recommendation_lists:
        for item in list(recs)[:k]:
            if item in bins:
                counts[bins[item]] += 1
    total = counts.sum()
    return counts / total if total else counts


def popularity_bin_recall(recommendation_lists: Sequence[Sequence[Any]], truths: Sequence[set[Any]],
                          bins: Mapping[Any, int], n_bins: int, k: int) -> np.ndarray:
    """Per bin: share of relevant (customer, item) pairs that were recommended. NaN for empty bins.

    Relevant items that are not in the training catalogue cannot be recommended by any model and are ignored.
    """
    hits, totals = np.zeros(n_bins), np.zeros(n_bins)
    for recs, truth in zip(recommendation_lists, truths, strict=True):
        top = set(list(recs)[:k])
        for item in truth:
            if item in bins:
                totals[bins[item]] += 1
                hits[bins[item]] += item in top
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, hits / totals, np.nan)


def oracle_revenue_at_k(relevant: Iterable[Any], prices: Mapping[Any, float], k: int) -> float:
    """Best possible ``revenue_at_k``: the k most valuable relevant items (unknown price counts as zero)."""
    values = sorted((prices.get(item, 0.0) for item in set(relevant)), reverse=True)
    return float(sum(values[:k]))
