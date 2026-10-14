"""Top-k ranking metrics with binary relevance, plus offline engagement and revenue proxies.

The proxies (hit rate, catalogue coverage, revenue) are computed on historical
purchases. They are offline stand-ins, not measurements of deployed engagement
or sales revenue.
"""
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


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
