"""Hyperparameter search for ALS on a validation window inside the training period."""
from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from .config import ALS_PARAMS
from .evaluate import evaluate
from .models import ALSModel
from .split import temporal_split

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
