"""Client-selection rules (Section 3.4 and the baselines of Section 5.1).

Every rule receives the same information and returns the indices of the
selected clients.  All factors are oriented so that *small is preferable*.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import Config


def factors(cfg: Config, e_hat, n_prev, rho, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eqs. (26)-(28), computed over the feasible set ``idx``."""
    e = e_hat[idx]
    e_t = (e - e.min()) / (e.max() - e.min() + cfg.eps0) + cfg.eps0
    u_t = (n_prev[idx] + 1.0) / (n_prev[idx].max() + 1.0)
    t_t = (1.0 - rho[idx]) + cfg.eps0
    return e_t, u_t, t_t


def product_score(cfg: Config, e_t, u_t, t_t, lam=None) -> np.ndarray:
    """Eq. (19): weighted geometric score (computed in the log domain)."""
    lE, lF, lT = lam if lam is not None else (cfg.lam_E, cfg.lam_F, cfg.lam_T)
    return np.exp(lE * np.log(e_t) + lF * np.log(u_t) + lT * np.log(t_t))


def additive_score(cfg: Config, e_t, u_t, t_t, lam=None) -> np.ndarray:
    """Weighted-sum scalarization of the same three factors."""
    lE, lF, lT = lam if lam is not None else (cfg.lam_E, cfg.lam_F, cfg.lam_T)
    return lE * e_t + lF * u_t + lT * t_t


def _smallest(values: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(values, kind="stable")[:k]


def select(rule: str, cfg: Config, feas: np.ndarray, k: int, e_hat, n_prev, rho,
           rng: np.random.Generator, state: Dict, lam=None, energy_scale: float = 1.0,
           shortlist: int | None = None) -> np.ndarray:
    idx = np.flatnonzero(feas)
    if len(idx) == 0:
        return idx
    k = min(k, len(idx))
    if rule == "all":
        return idx
    if rule == "random":
        return np.sort(rng.choice(idx, size=k, replace=False))
    if rule == "energy":
        return np.sort(idx[_smallest(e_hat[idx], k)])
    if rule == "fairfedcs":
        r = (state["a"] + 1.0) / (state["a"] + state["b"] + 2.0)
        psi = cfg.ffcs_sigma * r + state["Q"]
        return np.sort(idx[_smallest(-psi[idx], k)])
    e_t, u_t, t_t = factors(cfg, e_hat, n_prev, rho, idx)
    e_t = energy_scale * e_t                     # only used by the rescaling experiment
    if rule == "product":
        return np.sort(idx[_smallest(product_score(cfg, e_t, u_t, t_t, lam), k)])
    if rule == "additive":
        return np.sort(idx[_smallest(additive_score(cfg, e_t, u_t, t_t, lam), k)])
    if rule in ("seq_ef", "seq_fe"):
        m = min(len(idx), shortlist if shortlist is not None else cfg.seq_shortlist)
        first, second = (e_t, u_t) if rule == "seq_ef" else (u_t, e_t)
        keep = _smallest(first, m)
        return np.sort(idx[keep[_smallest(second[keep], k)]])
    if rule == "dpp":
        index = state["Z"][idx] - cfg.dpp_V * e_t          # drift-plus-penalty index
        return np.sort(idx[_smallest(-index, k)])
    raise ValueError(f"unknown selection rule: {rule}")


def init_state(P: int) -> Dict:
    return {"a": np.zeros(P), "b": np.zeros(P), "Q": np.zeros(P), "Z": np.zeros(P)}


def update_state(rule: str, cfg: Config, state: Dict, selected: np.ndarray, k_budget: int,
                 contrib: np.ndarray | None = None) -> None:
    """Bookkeeping of the queue-based baselines after a round."""
    P = len(state["Q"])
    x = np.zeros(P)
    x[selected] = 1.0
    eps = k_budget / P
    if rule == "fairfedcs":
        if contrib is not None:
            state["a"][selected] += (contrib >= 0)
            state["b"][selected] += (contrib < 0)
        r = (state["a"] + 1.0) / (state["a"] + state["b"] + 2.0)
        state["Q"] = np.maximum(0.0, state["Q"] + eps * r * (1.0 - x) - x)
    elif rule == "dpp":
        state["Z"] = np.maximum(0.0, state["Z"] + eps - x)
