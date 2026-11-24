"""Metrics and small statistical helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats


def jain(counts) -> float:
    """Jain index (sum n)^2 / (P sum n^2) of non-negative counts; 1 for all-zero input."""
    c = np.asarray(counts, float)
    s2 = float((c ** 2).sum())
    return 1.0 if s2 == 0 else float(c.sum() ** 2 / (len(c) * s2))


def expected_jain_uniform(P: int, k: int, R: int) -> float:
    """First-order expectation of the Jain index under uniform sampling without
    replacement of k of P clients in each of R rounds: mean Rk/P, variance
    R (k/P)(1 - k/P) per client."""
    mean = R * k / P
    var = R * (k / P) * (1 - k / P)
    return mean ** 2 / (mean ** 2 + var)


def mean_sd(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)), float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0


def paired_p(x, y) -> float:
    """Two-sided paired t-test over matched seeds; 1.0 when the differences are all zero."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return 1.0
    return float(stats.ttest_rel(x, y).pvalue)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def dump(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, cls=_Enc, indent=1))


def load(path):
    return json.loads(Path(path).read_text())
