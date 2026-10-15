"""Bayesian Personalised Ranking matrix factorisation (Rendle et al., 2009), mini-batch SGD in NumPy."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .models import _Base


def sample_negatives(rng: np.random.Generator, users: np.ndarray, n_items: int,
                     positive_keys: np.ndarray) -> np.ndarray:
    """Uniformly sampled items the customer has not bought.

    ``positive_keys`` is the sorted array of ``user * n_items + item`` for every purchase, or a flat boolean mask over
    all ``user * n_items + item`` keys (faster for large data). Draws that hit a purchased item are redrawn until none
    remain.
    """
    negatives = rng.integers(0, n_items, size=len(users))
    for _ in range(100):
        keys = users.astype(np.int64) * n_items + negatives
        if positive_keys.dtype == bool:
            clash = positive_keys[keys]
        else:
            pos = np.searchsorted(positive_keys, keys)
            clash = (pos < len(positive_keys)) & (positive_keys[np.minimum(pos, len(positive_keys) - 1)] == keys)
        if not clash.any():
            break
        negatives[clash] = rng.integers(0, n_items, size=int(clash.sum()))
    return negatives


def _scatter_add(target: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    """``target[index] += values`` with repeated indices summed (faster than ``np.add.at``)."""
    n = len(index)
    picker = sparse.csr_matrix((np.ones(n), (index, np.arange(n))), shape=(target.shape[0], n))
    target += picker @ values


class BPRModel(_Base):
    """Pairwise ranking loss ``-log sigmoid(x_ui - x_uj)`` with item biases and L2 regularisation.

    ``x_ui = p_u . q_i + b_i``. A mini-batch contains purchased pairs and one sampled negative each. A basket of
    products without a trained customer vector (new customers, the demo's picker) is scored with the mean of the
    basket's item factors; that is an approximation, not a training step.
    """

    name = "bpr"

    def __init__(self, factors: int = 64, epochs: int = 20, lr: float = 0.05, reg: float = 0.01,
                 batch_size: int = 2048, seed: int = 0):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

    def _fit(self) -> None:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = len(self.users), len(self.items)
        rows, cols = self.counts.nonzero()
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
        keys = np.zeros(n_users * n_items, dtype=bool)
        keys[rows.astype(np.int64) * n_items + cols] = True
        p = rng.normal(0.0, 0.1, (n_users, self.factors))
        q = rng.normal(0.0, 0.1, (n_items, self.factors))
        b = np.zeros(n_items)
        self.loss_history: list[float] = []
        lr, reg = self.lr, self.reg
        for _ in range(self.epochs):
            perm = rng.permutation(len(rows))
            total = 0.0
            for start in range(0, len(perm), self.batch_size):
                sel = perm[start : start + self.batch_size]
                u, i = rows[sel], cols[sel]
                j = sample_negatives(rng, u, n_items, keys)
                pu, qi, qj = p[u], q[i], q[j]
                x = (pu * (qi - qj)).sum(axis=1) + b[i] - b[j]
                g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))  # sigmoid(-x)
                total += float(np.logaddexp(0.0, -x).sum())
                gc = g[:, None]
                _scatter_add(p, u, lr * (gc * (qi - qj) - reg * pu))
                _scatter_add(q, i, lr * (gc * pu - reg * qi))
                _scatter_add(q, j, lr * (-gc * pu - reg * qj))
                _scatter_add(b, i, lr * (g - reg * b[i]))
                _scatter_add(b, j, lr * (-g - reg * b[j]))
            self.loss_history.append(total / len(rows))
        self.user_factors, self.item_factors, self.item_bias = p, q, b
        norms = np.linalg.norm(q, axis=1)
        norms[norms == 0] = 1.0
        self._unit = q / norms[:, None]

    def _scores_from_idx(self, idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(len(self.items))
        return np.asarray(self.item_factors @ self.item_factors[idx].mean(axis=0) + self.item_bias)

    def recommend(self, user: Any, k: int = 10) -> list[Any]:
        if user not in self.user_index:
            return self.ranking[:k]
        row = self.user_index[user]
        scores = self.item_factors @ self.user_factors[row] + self.item_bias
        return self._top(scores, self._seen_idx(user), k)

    def _contributions(self, seen_idx: np.ndarray, target: int) -> np.ndarray | None:
        return self._unit[seen_idx] @ self._unit[target]
