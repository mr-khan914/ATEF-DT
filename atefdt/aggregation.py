"""Aggregation rules: FedAvg, the reputation-weighted rule of ATEF-DT (Eq. (32)),
and the robust-aggregation baselines."""
from __future__ import annotations

import numpy as np

from .trust import geometric_median


def fedavg_weights(N: np.ndarray) -> np.ndarray:
    """Eq. (21): data-volume weights over the selected set."""
    return N / N.sum()


def clip_factors(U: np.ndarray) -> np.ndarray:
    """Section 3.6: scale every received update to at most the median norm."""
    norms = np.linalg.norm(U, axis=1)
    tau = np.median(norms)
    return np.minimum(1.0, tau / np.maximum(norms, 1e-300))


def reputation_coefficients(U: np.ndarray, rho_new: np.ndarray, rho_min: float):
    """Eqs. (24)-(25).  Returns (coefficients a_p = omega_p * c_p, accepted mask).

    The applied step is sum_p a_p * U_p.  The reputation weights omega_p sum to
    one over the accepted set; the coefficients sum to s <= 1 because of clipping.
    """
    accepted = rho_new >= rho_min
    a = np.zeros(len(U))
    if accepted.any():
        omega = rho_new[accepted] / rho_new[accepted].sum()
        a[accepted] = omega * clip_factors(U)[accepted]
    return a, accepted


def cw_median(U: np.ndarray) -> np.ndarray:
    return np.median(U, axis=0)


def trimmed_mean(U: np.ndarray, beta: float) -> np.ndarray:
    n = len(U)
    t = int(np.floor(beta * n))
    if t == 0 or n - 2 * t <= 0:
        return U.mean(axis=0)
    return np.sort(U, axis=0)[t:n - t].mean(axis=0)


def multi_krum(U: np.ndarray, f: int | None = None) -> np.ndarray:
    n = len(U)
    if f is None:
        f = max(0, (n - 3) // 2)
    if n <= 2 * f + 2 or n < 3:
        return U.mean(axis=0)
    sq = ((U[:, None, :] - U[None, :, :]) ** 2).sum(axis=2)
    nb = n - f - 2
    scores = np.sort(sq, axis=1)[:, 1:nb + 1].sum(axis=1)     # skip the zero self-distance
    return U[np.argsort(scores, kind="stable")[:n - f]].mean(axis=0)


def gm_aggregate(U: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
    return geometric_median(U, tol, max_iter)[0]
