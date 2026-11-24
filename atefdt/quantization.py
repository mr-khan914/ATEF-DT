"""Uniform stochastic quantization and the per-client bit-width rule (Section 3.5)."""
from __future__ import annotations

import numpy as np

from .config import Config

FULL_PRECISION_BITS = 32


def quantize(v: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """Unbiased b-bit uniform quantizer with a per-payload scale and offset.

    The payload is ``b`` bits per coordinate plus ``c_q`` bits of metadata
    (the minimum and the maximum of the vector in single precision).
    """
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return v.copy()
    levels = (1 << int(b)) - 1
    x = (v - lo) / (hi - lo) * levels
    fl = np.floor(x)
    q = fl + (rng.random(v.shape) < (x - fl))
    return lo + q * (hi - lo) / levels


def stability_surrogate(norm_last: float, norm_first: float) -> float:
    """Eq. (23): psi^(r) = ||w^(r) - w^(r-1)|| / ||w^(1) - w^(0)||; psi = 1 until both exist."""
    if norm_first <= 0.0 or norm_last < 0.0:
        return 1.0
    return float(norm_last / norm_first)


def bitwidth(cfg: Config, psi: float, h, h_ref: float, C, policy: str = "perclient"):
    """Eqs. (15)-(16).  ``policy``: perclient | global | fixed16 | fixed8 | full."""
    h = np.asarray(h, float)
    if policy == "full":
        return np.full(h.shape, FULL_PRECISION_BITS, dtype=int)
    if policy.startswith("fixed"):
        return np.full(h.shape, int(policy[5:]), dtype=int)
    base = cfg.b_max * min(psi, 1.0) ** cfg.gamma_b
    if policy == "global":
        g = np.ones_like(h)
    elif policy == "perclient":
        g = np.minimum(1.0, np.sqrt(h / h_ref)) * np.minimum(1.0, np.asarray(C, float) / cfg.C_ref)
    else:
        raise ValueError(f"unknown bit-width policy: {policy}")
    return np.ceil(np.clip(base * g, cfg.b_min, cfg.b_max) - 1e-12).astype(int)


def payload_bits(cfg: Config, b, d: int):
    """Bits on air for one update: b*d payload plus metadata (none at full precision)."""
    b = np.asarray(b)
    return b * d + np.where(b >= FULL_PRECISION_BITS, 0, cfg.c_q)


def compression_factor(cfg: Config, mean_b: float, d: int) -> float:
    """Corollary 1: 32 / (mean bit-width + c_q / d)."""
    return FULL_PRECISION_BITS / (mean_b + cfg.c_q / d)
