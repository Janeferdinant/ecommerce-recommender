"""Hyperparameter search for ALS on a validation window inside the training period."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import pandas as pd

from .config import ALS_PARAMS
from .evaluate import evaluate
from .evaluation import evaluate_extended
from .models import ALSModel
from .split import temporal_split, validation_split_by_protocol

DEFAULT_GRID: dict[str, list[float | int]] = {
    "factors": [16, 32, 64],
    "reg": [0.01, 0.1, 1.0],
    "iterations": [10, 15],
}


def validation_split(train: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the *training* data in time. The test period is never touched."""
    return temporal_split(train, val_fraction)


def tune_als(train: pd.DataFrame, grid: dict[str, list[Any]] | None = None, seed: int = 0, k: int = 10,
             val_fraction: float = 0.2) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the ALS settings with the best validation NDCG@k.

    Returns ``(best_params, table)``; ``table`` has one row per grid point (grid
    order). Ties go to the earlier grid point. ``alpha`` is held at its default.
    """
    grid = grid or DEFAULT_GRID
    fit, val = validation_split(train, val_fraction)
    names = list(grid)
    table: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values, strict=True))
        merged = {**ALS_PARAMS, **params}
        result = evaluate((fit, val), seed=seed, k=k, models=[ALSModel(seed=seed, **merged)])
        metrics = result["models"]["als"]
        table.append({**params, f"ndcg@{k}": metrics[f"ndcg@{k}"], f"precision@{k}": metrics[f"precision@{k}"],
                      f"recall@{k}": metrics[f"recall@{k}"], "n_val_users": result["n_eval_users"]})
    best_row = max(table, key=lambda row: row[f"ndcg@{k}"])  # max keeps the first of equal scores
    return {n: best_row[n] for n in names}, table


def tune_model(builder: Callable[..., Any], grid: dict[str, list[Any]], train: pd.DataFrame, protocol: str = "temporal",
               objective: str = "ndcg@10", k: int = 10, val_fraction: float = 0.2,
               seed: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grid search for any model on a validation window carved out of the *training* data only.

    ``builder(**params)`` returns an unfitted model. The window follows the dataset's protocol (a later slice of time,
    or each customer's last training interaction). The best row maximises ``objective`` (``ndcg@k`` or
    ``revenue@k``); ties go to the earlier grid point. Returns ``(best_params, table)``.
    """
    fit, val = validation_split_by_protocol(train, protocol, val_fraction)
    names = list(grid)
    table: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values, strict=True))
        result = evaluate_extended((fit, val), [builder(**params)], k=k, seed=seed, beyond=False)
        metrics = next(iter(result.summary.values()))
        row: dict[str, Any] = dict(params)
        for key in (f"ndcg@{k}", f"hit_rate@{k}", f"precision@{k}", f"recall@{k}", f"revenue@{k}"):
            if key in metrics:
                row[key] = metrics[key]
        row["n_val_users"] = len(result.users)
        table.append(row)
    if any(objective not in row for row in table):
        raise ValueError(f"objective {objective!r} is not available for this data")
    best_row = max(table, key=lambda row: row[objective])  # max keeps the first of equal scores
    return {n: best_row[n] for n in names}, table
