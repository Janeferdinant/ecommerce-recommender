"""The eight models of the study: how to build them, their tuning grids and the tuned-settings file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bpr import BPRModel
from .linear import EASEModel
from .models import ALSModel, ItemKNNModel, PopularityModel
from .neighbours import CoPurchaseModel, RecencyItemKNNModel

MODEL_ORDER = ["popularity", "itemknn", "itemknn_recency", "copurchase", "copurchase_price", "als", "ease", "bpr"]

LABELS = {
    "popularity": "Popularity",
    "itemknn": "Item-based CF",
    "itemknn_recency": "Item-based CF, recency-weighted",
    "copurchase": "Co-purchase (basket co-occurrence)",
    "copurchase_price": "Co-purchase + price-aware re-ranking",
    "als": "Implicit ALS",
    "ease": "EASE",
    "bpr": "BPR matrix factorisation",
}

# Tunable models: grid, objective on the validation window, and the settings used when nothing was tuned.
GRIDS: dict[str, dict[str, Any]] = {
    "itemknn_recency": {
        "grid": {"half_life_days": [7, 30, 90, 180, 365, None]},
        "objective": "ndcg@10",
        "defaults": {"half_life_days": 90},
    },
    "copurchase": {
        "grid": {"beta": [0.0, 0.25, 0.5, 0.75], "min_support": [1, 2, 3, 5]},
        "objective": "ndcg@10",
        "defaults": {"beta": 0.5, "min_support": 2},
    },
    # beta and min_support are inherited from the tuned "copurchase" model; only the price exponent is searched
    "copurchase_price": {
        "grid": {"gamma": [0.0, 0.25, 0.5, 1.0, 1.5]},
        "objective": "revenue@10",
        "defaults": {"beta": 0.5, "min_support": 2, "gamma": 0.5},
    },
    "als": {
        "grid": {"factors": [16, 32, 64], "reg": [0.01, 0.1, 1.0], "iterations": [10, 15]},
        "objective": "ndcg@10",
        "defaults": {"factors": 64, "iterations": 15, "reg": 1.0, "alpha": 10.0},
    },
    "ease": {
        "grid": {"reg": [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]},
        "objective": "ndcg@10",
        "defaults": {"reg": 500.0},
    },
    "bpr": {
        "grid": {"factors": [32, 64], "lr": [0.02, 0.05], "reg": [0.001, 0.01]},
        "objective": "ndcg@10",
        "defaults": {"factors": 64, "lr": 0.05, "reg": 0.01, "epochs": 20, "batch_size": 2048},
    },
}

SEEDED = {"als", "bpr"}


def make_model(name: str, params: dict[str, Any], seed: int = 0) -> Any:
    """Build an unfitted model by name; ``seed`` only reaches the models that use randomness."""
    if name == "popularity":
        return PopularityModel()
    if name == "itemknn":
        return ItemKNNModel()
    if name == "itemknn_recency":
        return RecencyItemKNNModel(**params)
    if name in ("copurchase", "copurchase_price"):
        return CoPurchaseModel(**params)
    if name == "als":
        return ALSModel(seed=seed, **params)
    if name == "ease":
        return EASEModel(**params)
    if name == "bpr":
        return BPRModel(seed=seed, **params)
    raise KeyError(name)


def build_models(tuned: dict[str, Any], seed: int = 0, has_price: bool = True) -> list[Any]:
    """All models in ``MODEL_ORDER`` with tuned settings (price-aware model skipped without prices)."""
    models = []
    for name in MODEL_ORDER:
        if name == "copurchase_price" and not has_price:
            continue
        params = tuned.get("models", {}).get(name, {}).get("params", GRIDS.get(name, {}).get("defaults", {}))
        models.append(make_model(name, dict(params), seed))
    return models


def write_tuned(path: Path, tuned: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_tuned(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return result
