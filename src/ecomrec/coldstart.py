"""New-customer (cold start) evaluation.

A random sample of the evaluation customers is treated as new: all of their training purchases except the first ``n``
(by time, then item id) are removed and every model is refitted on that reduced training set. Each model then
recommends from the ``n`` visible products only, through its basket mode. The ground truth is unchanged (products the
customer had not bought in the full training period and bought in the test period). A full-history fit gives the
reference for the same customers.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import item_prices, relevant_items
from .metrics import hit_rate_at_k, ndcg_at_k, recall_at_k, revenue_at_k


def truncate_histories(train: pd.DataFrame, users: Iterable[Any], n: int) -> pd.DataFrame:
    """Keep only the first ``n`` distinct products (by time, then item id) of the listed customers."""
    chosen = set(users)
    mask = train["user"].isin(chosen)
    ordered = train[mask].sort_values(["user", "ts", "item"], kind="mergesort")
    first = ordered.drop_duplicates(["user", "item"])
    keep_pairs = first.groupby("user").head(n)[["user", "item"]]
    kept = ordered.merge(keep_pairs, on=["user", "item"], how="inner")
    merged = pd.concat([train[~mask], kept.reindex(columns=train.columns)])
    return merged.sort_index(kind="mergesort").reset_index(drop=True)


def cold_start_eval(split: tuple[pd.DataFrame, pd.DataFrame], factories: dict[str, Callable[[], Any]],
                    n_values: Sequence[int] = (1, 2, 3), k: int = 10, seed: int = 0,
                    cold_fraction: float = 0.25) -> dict[str, Any]:
    """Hit rate, NDCG, recall (and revenue with prices) for new customers with 1 to 3 purchases."""
    train, test = split
    relevant = relevant_items(train, test)
    eligible = sorted(relevant)
    rng = np.random.default_rng(seed)
    size = max(1, int(round(len(eligible) * cold_fraction)))
    users = sorted(rng.choice(eligible, size=size, replace=False).tolist())
    prices = item_prices(train) if "price" in train.columns else None
    first_items = {
        u: g.sort_values(["ts", "item"], kind="mergesort").drop_duplicates("item")["item"].tolist()
        for u, g in train[train["user"].isin(set(users))].groupby("user")
    }

    def score(recs: list[Any], user: Any) -> dict[str, float]:
        truth = relevant[user]
        out = {f"hit_rate@{k}": hit_rate_at_k(recs, truth, k), f"ndcg@{k}": ndcg_at_k(recs, truth, k),
               f"recall@{k}": recall_at_k(recs, truth, k)}
        if prices is not None:
            out[f"revenue@{k}"] = revenue_at_k(recs, truth, prices, k)
        return out

    per_user: dict[str, dict[Any, dict[str, np.ndarray]]] = {name: {} for name in factories}
    summary: dict[str, dict[Any, dict[str, float]]] = {name: {} for name in factories}

    def record(name: str, key: Any, rows: list[dict[str, float]]) -> None:
        arrays = {m: np.asarray([r[m] for r in rows]) for m in rows[0]}
        per_user[name][key] = arrays
        summary[name][key] = {m: float(v.mean()) for m, v in arrays.items()}

    for name, factory in factories.items():
        full = factory().fit(train)
        record(name, "full", [score(full.recommend(u, k=k), u) for u in users])
    for n in n_values:
        reduced = truncate_histories(train, users, n)
        for name, factory in factories.items():
            model = factory().fit(reduced)
            rows = [score(model.recommend_for_items(first_items[u][:n], k=k), u) for u in users]
            record(name, n, rows)
    return {"users": users, "n_cold_users": len(users), "k": k, "n_values": list(n_values),
            "per_user": per_user, "summary": summary}
