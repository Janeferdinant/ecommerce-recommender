"""Reference implementation of the demo's "pick products" mode, run on the exported numbers.

The page (docs/index.html) implements the same maths in JavaScript and checks itself against the outputs this module
produced when the demo data was built. Positions are indices into the exported catalogue.
"""
from __future__ import annotations

from typing import Any

import numpy as np

NEIGHBOUR_MODELS = ("itemknn", "itemknn_recency", "ease", "copurchase")
TOP_K = 10
N_CONTRIBUTORS = 3


def _signif(value: float) -> float:
    return float(f"{value:.4g}")


def top_neighbours(matrix: np.ndarray, n: int) -> list[list[list[float]]]:
    """For every row, its ``n`` largest positive entries as ``[column, value]`` (ties: lower column first)."""
    rows = []
    for row in matrix:
        order = np.lexsort((np.arange(len(row)), -row))[:n]
        rows.append([[int(j), _signif(float(row[j]))] for j in order if row[j] > 0])
    return rows


def _shares(sources: list[int], values: list[float]) -> list[list[float]]:
    """Top contributors with shares that sum to one (rounded to two decimals)."""
    pairs = [(s, v) for s, v in zip(sources, values, strict=True) if v > 0]
    pairs.sort(key=lambda sv: (-sv[1], sv[0]))
    pairs = pairs[:N_CONTRIBUTORS]
    total = sum(v for _, v in pairs)
    return [[int(s), round(v / total, 2)] for s, v in pairs] if total > 0 else []


def _order(scores: np.ndarray) -> list[int]:
    return [int(j) for j in np.lexsort((np.arange(len(scores)), -scores))]


def _neighbour_recs(lists: list[list[list[float]]], chosen: list[int], n: int, k: int,
                    weights: np.ndarray | None = None) -> list[dict[str, Any]]:
    scores = np.zeros(n)
    contrib: dict[int, dict[int, float]] = {}
    for c in chosen:
        for j, s in lists[c]:
            j = int(j)
            if j in chosen:
                continue
            scores[j] += s
            contrib.setdefault(j, {})[c] = s
    final = scores * weights if weights is not None else scores
    ranked = [j for j in _order(final) if final[j] > 0][:k]
    recs = []
    for j in ranked:
        sources = list(contrib[j])
        rec: dict[str, Any] = {"i": j, "w": _shares(sources, [contrib[j][c] for c in sources])}
        if weights is not None:
            rec["f"] = float(weights[j])
        recs.append(rec)
    used = set(ranked)
    for p in range(n):  # back-fill with the best sellers when neighbours run out
        if len(recs) >= k:
            break
        if p not in chosen and p not in used:
            rec = {"i": p, "w": []}
            if weights is not None:
                rec["f"] = float(weights[p])
            recs.append(rec)
    return recs


def _solve_fold_in(als: dict[str, Any], chosen: list[int]) -> np.ndarray:
    f = np.asarray(als["factors"])
    w = als["alpha"] * np.log1p(1.0)
    a = np.asarray(als["gram"]) + w * f[chosen].T @ f[chosen] + als["reg"] * np.eye(f.shape[1])
    b = (1.0 + w) * f[chosen].sum(axis=0)
    return np.linalg.solve(a, b)


def _cosine_contributions(factors: np.ndarray, chosen: list[int], target: int) -> list[list[float]]:
    unit = factors / np.maximum(np.linalg.norm(factors, axis=1, keepdims=True), 1e-12)
    return _shares(chosen, [float(unit[c] @ unit[target]) for c in chosen])


def recommend_from_exported(data: dict[str, Any], chosen: list[int], k: int = TOP_K) -> dict[str, list[dict[str, Any]]]:
    """Top-k recommendations of every exported model for a basket of catalogue positions.

    Each entry has ``i`` (position), ``w`` (up to three ``[source position, share]`` pairs) and, for the price-aware
    model, ``f`` (the price factor applied to the score).
    """
    basket = sorted(set(chosen))
    n = len(data["catalogue"])
    out: dict[str, list[dict[str, Any]]] = {
        "popularity": [{"i": p, "w": []} for p in range(n) if p not in basket][:k]
    }
    for model in NEIGHBOUR_MODELS:
        if model in data["lists"]:
            out[model] = _neighbour_recs(data["lists"][model], basket, n, k)
    if "price_weight" in data and "copurchase" in data["lists"]:
        out["copurchase_price"] = _neighbour_recs(data["lists"]["copurchase"], basket, n, k,
                                                  np.asarray(data["price_weight"], dtype=np.float64))
    if "als" in data:
        factors = np.asarray(data["als"]["factors"])
        scores = factors @ _solve_fold_in(data["als"], basket)
        scores[basket] = -np.inf
        ranked = [j for j in _order(scores) if np.isfinite(scores[j])][:k]
        out["als"] = [{"i": j, "w": _cosine_contributions(factors, basket, j)} for j in ranked]
    if "bpr" in data:
        factors = np.asarray(data["bpr"]["factors"])
        scores = factors @ factors[basket].mean(axis=0) + np.asarray(data["bpr"]["bias"])
        scores[basket] = -np.inf
        ranked = [j for j in _order(scores) if np.isfinite(scores[j])][:k]
        out["bpr"] = [{"i": j, "w": _cosine_contributions(factors, basket, j)} for j in ranked]
    return out
