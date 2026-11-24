"""Credibility engine: geometric median, cosine trust signal, reputation (Section 3.6)."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def geometric_median(U: np.ndarray, tol: float = 1e-5, max_iter: int = 200,
                     weights: np.ndarray | None = None) -> Tuple[np.ndarray, int]:
    """(Weighted) geometric median by the Weiszfeld iteration.

    Minimizes ``sum_p weights_p * ||U_p - z||``; the iteration stops when the
    relative change of the iterate is below ``tol``.
    """
    c = np.ones(len(U)) if weights is None else np.asarray(weights, float)
    if c.sum() <= 0:
        c = np.ones(len(U))
    z = (c[:, None] * U).sum(axis=0) / c.sum()
    for it in range(1, max_iter + 1):
        dist = np.maximum(np.linalg.norm(U - z, axis=1), 1e-12)
        wts = c / dist
        z_new = (wts[:, None] * U).sum(axis=0) / wts.sum()
        if np.linalg.norm(z_new - z) <= tol * max(np.linalg.norm(z), 1e-12):
            return z_new, it
        z = z_new
    return z, max_iter


def cosine(U: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Eq. (28): cosine between each received update (rows of U) and the reference."""
    nu = np.linalg.norm(U, axis=1)
    nr = np.linalg.norm(ref)
    out = np.zeros(len(U))
    ok = (nu > 0) & (nr > 0)
    out[ok] = (U[ok] @ ref) / (nu[ok] * nr)
    return out


def update_reputation(rho: np.ndarray, phi: np.ndarray, mu: float) -> np.ndarray:
    """Eqs. (20)-(21): rectify the trust signal, then exponential forgetting."""
    return mu * rho + (1.0 - mu) * np.maximum(phi, 0.0)


def exclusion_bound(rho0: float, rho_min: float, mu: float) -> int:
    """Lemma 1: first rejected submission of a client whose cosine is never positive."""
    return int(math.floor(math.log(rho_min / rho0) / math.log(mu))) + 1
