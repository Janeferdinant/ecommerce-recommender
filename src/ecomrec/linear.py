"""EASE: Embarrassingly Shallow Autoencoder (Steck, 2019), a closed-form linear item-item model."""
from __future__ import annotations

import numpy as np

from .models import _Base


class EASEModel(_Base):
    """Linear autoencoder with a zero diagonal, solved in closed form.

    With ``X`` the binary customer-by-product matrix and ``G = X'X + reg * I``, the weights that minimise
    ``||X - XB||^2 + reg * ||B||^2`` subject to ``diag(B) = 0`` are ``B = -G^-1 / diag(G^-1)`` (diagonal set to zero).
    The score of a product for a customer is the sum of ``B`` rows of the products they bought.
    """

    name = "ease"

    def __init__(self, reg: float = 500.0):
        if reg <= 0:
            raise ValueError("reg must be positive")
        self.reg = float(reg)

    def _fit(self) -> None:
        x = (self.counts > 0).astype(np.float64)
        gram = (x.T @ x).toarray()
        gram[np.diag_indices_from(gram)] += self.reg
        inverse = np.linalg.inv(gram)
        weights = -inverse / np.diag(inverse)
        weights[np.diag_indices_from(weights)] = 0.0
        self.weights = weights

    def _scores_from_idx(self, idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(len(self.items))
        return np.asarray(self.weights[idx].sum(axis=0))

    def _contributions(self, seen_idx: np.ndarray, target: int) -> np.ndarray | None:
        return self.weights[seen_idx, target]
