"""Item-neighbourhood models: recency-weighted item CF and a co-purchase baseline with business rules."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .models import ItemKNNModel, _Base

_DAY = pd.Timedelta(days=1)


class RecencyItemKNNModel(ItemKNNModel):
    """Item-based CF where each purchase is weighted by ``0.5 ** (age / half_life)``.

    Age is measured from the end of the training period. The weights apply twice: to the customers' purchase
    vectors when item-item cosine similarity is computed, and to a customer's own history when their score is
    summed. ``half_life_days=None`` switches the decay off and gives exactly ``ItemKNNModel``. A basket without
    timestamps (new customer, demo picker) is weighted equally.
    """

    name = "itemknn_recency"

    def __init__(self, half_life_days: float | None = 90.0):
        if half_life_days is not None and half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = half_life_days
        self.weights: sparse.csr_matrix | None = None

    def _fit(self) -> None:
        if self.half_life_days is None:
            super()._fit()
            self.weights = None
            return
        latest = self._train.groupby(["user", "item"])["ts"].max().reset_index()
        age = (self._train["ts"].max() - latest["ts"]) / _DAY
        weight = np.power(0.5, age.to_numpy(dtype=np.float64) / self.half_life_days)
        rows = latest["user"].map(self.user_index).to_numpy()
        cols = latest["item"].map(self.item_index).to_numpy()
        w = sparse.csr_matrix((weight, (rows, cols)), shape=self.counts.shape)
        w.sort_indices()
        self.weights = w
        co = (w.T @ w).toarray()
        norms = np.sqrt(np.diag(co))
        norms[norms == 0] = 1.0
        sim = co / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)
        self.similarity = sim.astype(np.float32)

    def _scores_from_idx(self, idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(len(self.items))
        if self.weights is None:
            return super()._scores_from_idx(idx, counts)
        return (self.similarity[idx] * counts[:, None]).sum(axis=0)

    def recommend(self, user: Any, k: int = 10) -> list[Any]:
        if self.weights is None:
            return super().recommend(user, k)
        if user not in self.user_index:
            return self.ranking[:k]
        row = self.user_index[user]
        lo, hi = self.weights.indptr[row], self.weights.indptr[row + 1]
        idx, w = self.weights.indices[lo:hi], self.weights.data[lo:hi]
        return self._top(self._scores_from_idx(idx, w), idx, k)


class CoPurchaseModel(_Base):
    """'Customers who bought this also bought' from basket co-occurrence, plus simple business rules.

    ``co[j, i]`` counts baskets that hold both products (a basket is one customer at one timestamp, or the
    ``basket`` column when present). The association is ``co / (n_j * n_i ** beta)`` where ``n`` counts baskets
    per product; ``beta`` damps best sellers. Rules:

    * ``min_support``: pairs seen in fewer baskets are ignored;
    * ``gamma``: price-aware re-ranking, scores are multiplied by ``(price / median price) ** gamma`` using
      training-period median unit prices (``gamma = 0`` switches the rule off; it needs a ``price`` column).
    """

    def __init__(self, beta: float = 0.5, min_support: int = 2, gamma: float = 0.0, name: str | None = None):
        if beta < 0 or min_support < 1 or gamma < 0:
            raise ValueError("beta and gamma must be non-negative and min_support at least 1")
        self.beta = beta
        self.min_support = min_support
        self.gamma = gamma
        self.name = name or ("copurchase_price" if gamma > 0 else "copurchase")

    def _fit(self) -> None:
        train = self._train
        if self.gamma > 0 and "price" not in train.columns:
            raise ValueError("the price rule needs a 'price' column")
        basket = train["basket"] if "basket" in train.columns else train["user"] + "|" + train["ts"].astype(str)
        pairs = pd.DataFrame({"basket": basket.to_numpy(), "item": train["item"].to_numpy()}).drop_duplicates()
        rows = pd.factorize(pairs["basket"])[0]
        cols = pairs["item"].map(self.item_index).to_numpy()
        matrix = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (rows, cols)),
                                   shape=(int(rows.max()) + 1, len(self.items)))
        co = (matrix.T @ matrix).toarray()
        self.basket_counts = np.diag(co).copy()
        self.co = co
        kept = np.where(co >= self.min_support, co, 0.0)
        n = np.maximum(self.basket_counts, 1.0)
        assoc = kept / n[:, None] / np.power(n, self.beta)[None, :]
        np.fill_diagonal(assoc, 0.0)
        self.assoc = assoc.astype(np.float32)
        if self.gamma > 0:
            median = train.groupby("item")["price"].median().reindex(self.items)
            ref = float(median.median())
            self.price_weight = np.power(median.to_numpy(dtype=np.float64) / ref, self.gamma)
        else:
            self.price_weight = np.ones(len(self.items))

    def cooccurrence(self, pair: tuple[Any, Any]) -> int:
        """Number of baskets that hold both products."""
        return int(self.co[self.item_index[pair[0]], self.item_index[pair[1]]])

    def scores_for_items(self, items: Any) -> np.ndarray:
        idx = np.array(sorted(self.item_index[i] for i in items), dtype=np.int64)
        return self._scores_from_idx(idx, np.ones(len(idx)))

    def _scores_from_idx(self, idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(len(self.items))
        return np.asarray(self.assoc[idx].sum(axis=0) * self.price_weight)

    def _contributions(self, seen_idx: np.ndarray, target: int) -> np.ndarray | None:
        return self.assoc[seen_idx, target]
