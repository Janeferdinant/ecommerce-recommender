"""Extended offline evaluation: accuracy, engagement and revenue proxies plus beyond-accuracy measures.

It reproduces ``ecomrec.evaluate.evaluate_detailed`` exactly for the shared metrics (a test checks this) and adds
novelty, intra-list diversity, popularity-bias curves, an oracle-normalised revenue proxy and a price-drift
sensitivity check. All numbers are offline measurements on historical data.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .evaluate import item_prices, relevant_items
from .metrics import (
    assign_popularity_bins,
    catalogue_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    jaccard_distance,
    ndcg_at_k,
    novelty_at_k,
    oracle_revenue_at_k,
    popularity_bin_recall,
    popularity_bin_shares,
    precision_at_k,
    recall_at_k,
    revenue_at_k,
)


@dataclass
class Evaluation:
    k: int
    users: list[Any]
    n_eligible: int
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    per_user: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    popularity_curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    recs: dict[str, list[list[Any]]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _cosine_matrix(counts: sparse.csr_matrix) -> np.ndarray:
    binary = (counts > 0).astype(np.float32)
    norms = np.sqrt(np.asarray(binary.multiply(binary).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    scaled = binary @ sparse.diags(1.0 / norms)
    return np.asarray((scaled.T @ scaled).toarray(), dtype=np.float32)


def _lookup(sim: np.ndarray, index: dict[Any, int]) -> Callable[[Any, Any], float]:
    return lambda a, b: float(sim[index[a], index[b]])


def evaluate_extended(split: tuple[pd.DataFrame, pd.DataFrame], models: list[Any], k: int = 10, seed: int = 0,
                      max_users: int | None = None, genres: dict[Any, frozenset[str]] | None = None,
                      beyond: bool = True, n_bins: int = 10) -> Evaluation:
    """Fit and evaluate every model on the same customers; ``beyond=False`` skips the slower measures."""
    train, test = split
    relevant = relevant_items(train, test)
    users = sorted(relevant)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())
    has_price = "price" in train.columns
    train_prices = item_prices(train) if has_price else {}
    test_prices = item_prices(test) if has_price else {}
    n_catalogue = int(train["item"].nunique())
    truths = [relevant[u] for u in users]

    result = Evaluation(k=k, users=users, n_eligible=len(relevant))
    result.meta = {"n_catalogue": n_catalogue, "n_eval_users": len(users), "n_eligible_users": len(relevant)}
    catalogue = set(train["item"].unique())
    pairs = sum(len(t) for t in truths)
    result.meta["unreachable_relevant_share"] = (
        sum(1 for t in truths for i in t if i not in catalogue) / pairs if pairs else 0.0
    )
    if has_price:
        ratios = [test_prices[i] / train_prices[i] for t in truths for i in t
                  if i in train_prices and i in test_prices and train_prices[i] > 0]
        result.meta["price_drift_median"] = float(np.median(ratios)) if ratios else 1.0

    def genre_similarity(a: Any, b: Any) -> float:
        assert genres is not None
        return 1.0 - jaccard_distance(genres.get(a, frozenset()), genres.get(b, frozenset()))

    similarity_fn: Callable[[Any, Any], float] | None = None
    for model in models:
        model.fit(train)
        if beyond and similarity_fn is None:
            similarity_fn = _lookup(_cosine_matrix(model.counts), model.item_index)
            buyers = dict(zip(model.items, model.buyers.tolist(), strict=True))
            n_users_train = len(model.users)
            bins = assign_popularity_bins(buyers, n_bins)
        names = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"]
        if has_price:
            names += [f"revenue@{k}", f"oracle_revenue@{k}", f"revenue_testprice@{k}"]
        if beyond:
            names += [f"novelty@{k}", f"diversity@{k}"] + ([f"genre_diversity@{k}"] if genres else [])
        values: dict[str, list[float]] = {n: [] for n in names}
        all_recs: list[list[Any]] = []
        for user, truth in zip(users, truths, strict=True):
            recs = model.recommend(user, k=k)
            all_recs.append(recs)
            values[f"precision@{k}"].append(precision_at_k(recs, truth, k))
            values[f"recall@{k}"].append(recall_at_k(recs, truth, k))
            values[f"ndcg@{k}"].append(ndcg_at_k(recs, truth, k))
            values[f"hit_rate@{k}"].append(hit_rate_at_k(recs, truth, k))
            if has_price:
                values[f"revenue@{k}"].append(revenue_at_k(recs, truth, train_prices, k))
                values[f"oracle_revenue@{k}"].append(oracle_revenue_at_k(truth, train_prices, k))
                values[f"revenue_testprice@{k}"].append(revenue_at_k(recs, truth, test_prices, k))
            if beyond and similarity_fn is not None:
                values[f"novelty@{k}"].append(novelty_at_k(recs, buyers, n_users_train, k))
                values[f"diversity@{k}"].append(
                    intra_list_diversity(recs, similarity_fn, k))
                if genres:
                    values[f"genre_diversity@{k}"].append(
                        intra_list_diversity(recs, genre_similarity, k))
        result.per_user[model.name] = {n: np.asarray(v) for n, v in values.items()}
        summary = {n: float(np.mean(v)) for n, v in values.items() if n != f"oracle_revenue@{k}"}
        summary[f"coverage@{k}"] = catalogue_coverage(all_recs, n_catalogue)
        if has_price:
            oracle = float(np.mean(values[f"oracle_revenue@{k}"]))
            summary[f"revenue_capture@{k}"] = summary[f"revenue@{k}"] / oracle if oracle > 0 else 0.0
        result.summary[model.name] = summary
        result.recs[model.name] = all_recs
        if beyond:
            result.popularity_curves[model.name] = {
                "rec_share": popularity_bin_shares(all_recs, bins, n_bins, k).tolist(),
                "recall": popularity_bin_recall(all_recs, truths, bins, n_bins, k).tolist(),
            }
    return result
