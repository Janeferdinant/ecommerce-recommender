"""Offline evaluation of all models on a temporal split.

Besides Precision, Recall and NDCG, the harness computes three offline proxies:
hit rate and catalogue coverage (engagement proxies) and Revenue@k (revenue
proxy). They are measured on historical purchases and are not measurements of
live engagement or revenue.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ALS_PARAMS
from .metrics import (
    catalogue_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    revenue_at_k,
)
from .models import ALSModel, ItemKNNModel, PopularityModel

Split = tuple[pd.DataFrame, pd.DataFrame]


def build_models(seed: int = 0, als_params: dict[str, Any] | None = None) -> list[Any]:
    """Popularity, item-based CF and ALS (tuned defaults from ``ecomrec.config``)."""
    params: dict[str, Any] = dict(ALS_PARAMS)
    params.update(als_params or {})
    return [PopularityModel(), ItemKNNModel(), ALSModel(seed=seed, **params)]


def relevant_items(train: pd.DataFrame, test: pd.DataFrame) -> dict[Any, set[Any]]:
    """Test-period purchases that are new to the customer (not bought in training)."""
    seen = train.groupby("user")["item"].apply(set).to_dict()
    bought = test.groupby("user")["item"].apply(set).to_dict()
    out = {}
    for user, items in bought.items():
        if user in seen:
            fresh = items - seen[user]
            if fresh:
                out[user] = fresh
    return out


def item_prices(train: pd.DataFrame) -> dict[Any, float]:
    """Median unit price per item in the *training* period (never the test period)."""
    return train.groupby("item")["price"].median().to_dict()


def evaluate_detailed(split: Split, seed: int = 0, k: int = 10, als_params: dict[str, Any] | None = None,
                      max_users: int | None = None, models: list[Any] | None = None):
    """Evaluate every model; return ``(summary, per_user)``.

    ``per_user[model][metric]`` is an array with one value per evaluation
    customer (same customer order for every model), which the bootstrap uses.
    Evaluation customers are customers seen in training who bought at least one
    new item in the test period; ``max_users`` subsamples them with ``seed``.
    """
    train, test = split
    relevant = relevant_items(train, test)
    users = sorted(relevant)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())
    prices = item_prices(train) if "price" in train.columns else None
    n_catalogue = train["item"].nunique()
    models = models if models is not None else build_models(seed, als_params)

    summary_models: dict[str, dict[str, float]] = {}
    per_user: dict[str, dict[str, np.ndarray]] = {}
    for model in models:
        model.fit(train)
        values: dict[str, list[float]] = {
            f"precision@{k}": [], f"recall@{k}": [], f"ndcg@{k}": [], f"hit_rate@{k}": [],
        }
        if prices is not None:
            values[f"revenue@{k}"] = []
        all_recs = []
        for user in users:
            recs = model.recommend(user, k=k)
            all_recs.append(recs)
            truth = relevant[user]
            values[f"precision@{k}"].append(precision_at_k(recs, truth, k))
            values[f"recall@{k}"].append(recall_at_k(recs, truth, k))
            values[f"ndcg@{k}"].append(ndcg_at_k(recs, truth, k))
            values[f"hit_rate@{k}"].append(hit_rate_at_k(recs, truth, k))
            if prices is not None:
                values[f"revenue@{k}"].append(revenue_at_k(recs, truth, prices, k))
        per_user[model.name] = {name: np.asarray(v) for name, v in values.items()}
        summary_models[model.name] = {name: float(np.mean(v)) for name, v in values.items()}
        summary_models[model.name][f"coverage@{k}"] = catalogue_coverage(all_recs, n_catalogue)
    summary = {
        "k": k,
        "seed": seed,
        "n_eval_users": len(users),
        "n_eligible_users": len(relevant),
        "n_catalogue": int(n_catalogue),
        "models": summary_models,
    }
    return summary, per_user


def evaluate(split: Split, seed: int = 0, k: int = 10, als_params: dict[str, Any] | None = None,
             max_users: int | None = None, models: list[Any] | None = None) -> dict[str, Any]:
    """Per-model Precision, Recall, NDCG, hit rate, coverage and (with prices) Revenue at k."""
    return evaluate_detailed(split, seed, k, als_params, max_users, models)[0]
