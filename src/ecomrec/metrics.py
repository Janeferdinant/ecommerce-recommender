"""Top-k ranking metrics with binary relevance."""
import math
from typing import Iterable, Sequence, Set


def _hits(recommended: Sequence, relevant: Set, k: int) -> int:
    return sum(1 for item in list(recommended)[:k] if item in relevant)


def precision_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Share of the top-k slots that hold a relevant item (always divided by k)."""
    relevant = set(relevant)
    return _hits(recommended, relevant, k) / k


def recall_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Share of the relevant items found in the top k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return _hits(recommended, relevant, k) / len(relevant)


def ndcg_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
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
