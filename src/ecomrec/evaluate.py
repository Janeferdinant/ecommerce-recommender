"""Offline evaluation of all models on a temporal split."""
from __future__ import annotations

import numpy as np

from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .models import ALSModel, ItemKNNModel, PopularityModel


def build_models(seed: int = 0, als_params: dict | None = None):
    params = {"factors": 32, "iterations": 15, "reg": 0.1, "alpha": 10.0}
    params.update(als_params or {})
    return [PopularityModel(), ItemKNNModel(), ALSModel(seed=seed, **params)]


def relevant_items(train, test) -> dict:
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


def evaluate(split, seed: int = 0, k: int = 10, als_params: dict | None = None,
             max_users: int | None = None, models=None) -> dict:
    """Return per-model Precision@k, Recall@k and NDCG@k.

    Evaluation users are customers seen in training who bought at least one
    new item in the test period. If ``max_users`` is set, customers are
    subsampled with ``seed``.
    """
    train, test = split
    relevant = relevant_items(train, test)
    users = sorted(relevant)
    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())
    models = models if models is not None else build_models(seed, als_params)
    results = {}
    for model in models:
        model.fit(train)
        p, r, n = [], [], []
        for user in users:
            recs = model.recommend(user, k=k)
            p.append(precision_at_k(recs, relevant[user], k))
            r.append(recall_at_k(recs, relevant[user], k))
            n.append(ndcg_at_k(recs, relevant[user], k))
        results[model.name] = {
            f"precision@{k}": float(np.mean(p)),
            f"recall@{k}": float(np.mean(r)),
            f"ndcg@{k}": float(np.mean(n)),
        }
    return {
        "k": k,
        "seed": seed,
        "n_eval_users": len(users),
        "n_eligible_users": len(relevant),
        "models": results,
    }
