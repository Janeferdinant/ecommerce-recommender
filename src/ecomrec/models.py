"""Three recommenders behind one small interface.

* ``PopularityModel``: most distinct buyers in the training period.
* ``ItemKNNModel``: item-based collaborative filtering with cosine similarity.
* ``ALSModel``: implicit-feedback matrix factorisation by alternating least
  squares (Hu, Koren and Volinsky, 2008), written in NumPy.

All three share the same contract: ``fit(train)`` takes a DataFrame with the
columns ``user``, ``item`` and ``ts``; ``recommend(user, k)`` returns item ids,
best first, and never returns an item the user bought in the training data.
Ties are broken by popularity and then by item order, so results are fully
deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class _Base:
    name = "base"

    # ---- fitting ---------------------------------------------------------
    def fit(self, train: pd.DataFrame):
        self.users = sorted(train["user"].unique())
        self.items = sorted(train["item"].unique())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.item_index = {it: i for i, it in enumerate(self.items)}
        pairs = train.groupby(["user", "item"]).size().reset_index(name="n")
        rows = pairs["user"].map(self.user_index).to_numpy()
        cols = pairs["item"].map(self.item_index).to_numpy()
        self.counts = sparse.csr_matrix(
            (pairs["n"].to_numpy(dtype=np.float64), (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )
        self.counts.sort_indices()
        self.buyers = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        self._order_key = np.lexsort((np.arange(len(self.items)), -self.buyers))
        self.ranking = [self.items[i] for i in self._order_key]
        self._fit()
        return self

    def _fit(self):
        pass

    # ---- lookups ---------------------------------------------------------
    def _seen_idx(self, user) -> np.ndarray:
        row = self.user_index[user]
        return self.counts.indices[self.counts.indptr[row] : self.counts.indptr[row + 1]]

    def seen_items(self, user) -> list:
        return [self.items[i] for i in self._seen_idx(user)]

    def count(self, user, item) -> float:
        return float(self.counts[self.user_index[user], self.item_index[item]])

    # ---- scoring ---------------------------------------------------------
    def _scores_from_idx(self, idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _top(self, scores: np.ndarray, exclude: np.ndarray, k: int) -> list:
        scores = np.array(scores, dtype=np.float64)
        scores[exclude] = -np.inf
        # primary: score desc; then popularity desc; then item order
        order = np.lexsort((np.arange(len(scores)), -self.buyers, -scores))
        return [self.items[i] for i in order[:k] if np.isfinite(scores[i])]

    def recommend(self, user, k: int = 10) -> list:
        if user not in self.user_index:
            return self.ranking[:k]
        row = self.user_index[user]
        lo, hi = self.counts.indptr[row], self.counts.indptr[row + 1]
        idx, cnt = self.counts.indices[lo:hi], self.counts.data[lo:hi]
        return self._top(self._scores_from_idx(idx, cnt), idx, k)

    def recommend_for_items(self, items, k: int = 10) -> list:
        """Recommend for an ad-hoc basket of item ids (the demo's picker mode)."""
        idx = np.array(sorted(self.item_index[i] for i in items), dtype=np.int64)
        return self._top(self._scores_from_idx(idx, np.ones(len(idx))), idx, k)

    def explain(self, user, item):
        """The purchased item that best 'explains' the recommendation, or None."""
        return None

    def _best_source(self, seen_idx: np.ndarray, target: int, sim_row: np.ndarray):
        if len(seen_idx) == 0:
            return None
        sims = sim_row[seen_idx]
        best = int(np.argmax(sims))
        return self.items[seen_idx[best]] if sims[best] > 0 else None


class PopularityModel(_Base):
    name = "popularity"

    def _scores_from_idx(self, idx, counts):
        return self.buyers.astype(np.float64)


class ItemKNNModel(_Base):
    """Item-item cosine similarity on binary purchase vectors."""

    name = "itemknn"

    def _fit(self):
        binary = (self.counts > 0).astype(np.float32)
        co = (binary.T @ binary).toarray()
        norms = np.sqrt(np.diag(co))
        norms[norms == 0] = 1.0
        sim = co / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)
        self.similarity = sim.astype(np.float32)

    def _scores_from_idx(self, idx, counts):
        if len(idx) == 0:
            return np.zeros(len(self.items))
        return self.similarity[idx].sum(axis=0)

    def explain(self, user, item):
        j = self.item_index[item]
        return self._best_source(self._seen_idx(user), j, self.similarity[:, j])


class ALSModel(_Base):
    """Implicit-feedback ALS.

    Preference is 1 for every purchased item. Confidence is
    ``1 + alpha * log1p(n)`` where ``n`` is the number of baskets that held the
    item. The loop ends with the user step, so user factors are consistent with
    the final item factors and ``fold_in`` reproduces them exactly.
    """

    name = "als"

    def __init__(self, factors: int = 32, iterations: int = 15, reg: float = 0.1,
                 alpha: float = 10.0, seed: int = 0):
        self.factors = factors
        self.iterations = iterations
        self.reg = reg
        self.alpha = alpha
        self.seed = seed

    def _solve_row(self, fixed, fixed_gram, idx, strength):
        f = fixed[idx]
        w = self.alpha * strength
        a = fixed_gram + (f.T * w) @ f + self.reg * np.eye(self.factors)
        b = (1.0 + w) @ f
        return np.linalg.solve(a, b)

    def _solve_side(self, ratings: sparse.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        gram = fixed.T @ fixed
        out = np.zeros((ratings.shape[0], self.factors))
        for r in range(ratings.shape[0]):
            lo, hi = ratings.indptr[r], ratings.indptr[r + 1]
            if hi > lo:
                out[r] = self._solve_row(fixed, gram, ratings.indices[lo:hi], ratings.data[lo:hi])
        return out

    def _fit(self):
        rng = np.random.default_rng(self.seed)
        strength = self.counts.copy()
        strength.data = np.log1p(strength.data)
        strength_t = strength.T.tocsr()
        strength_t.sort_indices()
        self.user_factors = rng.normal(0.0, 0.1, (len(self.users), self.factors))
        for _ in range(self.iterations):
            self.item_factors = self._solve_side(strength_t, self.user_factors)
            self.user_factors = self._solve_side(strength, self.item_factors)
        self.gram = self.item_factors.T @ self.item_factors
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1.0
        self._unit = self.item_factors / norms[:, None]

    def fold_in(self, item_idx, counts) -> np.ndarray:
        """User vector for an ad-hoc basket (same maths as the last training step)."""
        idx = np.asarray(item_idx, dtype=np.int64)
        return self._solve_row(self.item_factors, self.gram, idx, np.log1p(np.asarray(counts, dtype=np.float64)))

    def _scores_from_idx(self, idx, counts):
        if len(idx) == 0:
            return np.zeros(len(self.items))
        return self.item_factors @ self.fold_in(idx, counts)

    def recommend(self, user, k: int = 10) -> list:
        if user not in self.user_index:
            return self.ranking[:k]
        row = self.user_index[user]
        return self._top(self.item_factors @ self.user_factors[row], self._seen_idx(user), k)

    def explain(self, user, item):
        j = self.item_index[item]
        return self._best_source(self._seen_idx(user), j, self._unit @ self._unit[j])
