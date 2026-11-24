"""Two-layer perceptron with a flat parameter vector (pure NumPy)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


class MLP:
    """in_dim -> hidden (ReLU) -> 1 (logit); binary cross-entropy loss.

    For WDBC (in_dim = 30, hidden = 64) the parameter count is
    30*64 + 64 + 64 + 1 = 2049.
    """

    def __init__(self, in_dim: int, hidden: int):
        self.in_dim, self.hidden = in_dim, hidden
        self.d = in_dim * hidden + hidden + hidden + 1
        a = in_dim * hidden
        self._s = (slice(0, a), slice(a, a + hidden), slice(a + hidden, a + 2 * hidden))

    def init(self, rng: np.random.Generator) -> np.ndarray:
        w = np.zeros(self.d)
        lim1 = np.sqrt(6.0 / self.in_dim)              # He-uniform
        lim2 = np.sqrt(6.0 / self.hidden)
        w[self._s[0]] = rng.uniform(-lim1, lim1, self.in_dim * self.hidden)
        w[self._s[2]] = rng.uniform(-lim2, lim2, self.hidden)
        return w

    def _unpack(self, w: np.ndarray):
        W1 = w[self._s[0]].reshape(self.hidden, self.in_dim)
        return W1, w[self._s[1]], w[self._s[2]], w[-1]

    def logits(self, w: np.ndarray, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self._unpack(w)
        return np.maximum(X @ W1.T + b1, 0.0) @ W2 + b2

    def loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        z = self.logits(w, X)
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def accuracy(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0:
            return float("nan")
        return float(np.mean((self.logits(w, X) > 0) == (y > 0.5)))

    def loss_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Mean loss over the batch and its gradient (one pass over the batch)."""
        W1, b1, W2, b2 = self._unpack(w)
        Z = X @ W1.T + b1
        Hh = np.maximum(Z, 0.0)
        z = Hh @ W2 + b2
        n = len(y)
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        dz = (1.0 / (1.0 + np.exp(-z)) - y) / n
        g = np.empty(self.d)
        dZ = np.outer(dz, W2) * (Z > 0)
        g[self._s[0]] = (dZ.T @ X).ravel()
        g[self._s[1]] = dZ.sum(axis=0)
        g[self._s[2]] = Hh.T @ dz
        g[-1] = dz.sum()
        return loss, g
