"""Data: the WDBC benchmark, a fixed split, and label-skewed client partitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import Config


@dataclass
class Split:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xval: np.ndarray
    yval: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray


@dataclass
class Partition:
    train_idx: List[np.ndarray]     # indices into the training split, one array per client
    test_idx: List[np.ndarray]      # indices into the test split, one array per client
    props: np.ndarray               # (classes, P) class-wise client proportions

    @property
    def sizes(self) -> np.ndarray:
        return np.array([len(i) for i in self.train_idx])


def load_split(cfg: Config) -> Split:
    """WDBC (569 samples, 30 features); label 1 = malignant.

    The split depends only on ``cfg.split_seed`` and is therefore identical in
    every run.  Features are standardized with training-split statistics only.
    """
    ds = load_breast_cancer()
    X = ds.data.astype(np.float64)
    y = (ds.target == 0).astype(np.float64)          # scikit-learn codes malignant as 0
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_frac, stratify=y, random_state=cfg.split_seed)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr, ytr, test_size=cfg.val_frac, stratify=ytr, random_state=cfg.split_seed)
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    std[std == 0] = 1.0
    scale = lambda A: (A - mean) / std
    return Split(scale(Xtr), ytr, scale(Xval), yval, scale(Xte), yte)


def dirichlet_partition(y: np.ndarray, yte: np.ndarray, P: int, alpha: float,
                        min_size: int, rng: np.random.Generator) -> Partition:
    """Label-skewed partition: class c is spread over clients with Dir(alpha) proportions.

    The draw is repeated until every client holds at least ``min_size`` training
    samples.  Test samples are attached to clients with the same class-wise
    proportions, so each client has a local test subset that mirrors its
    training distribution (used only for the group-level evaluation).
    """
    classes = np.unique(y)
    for _ in range(10_000):
        props = rng.dirichlet(alpha * np.ones(P), size=len(classes))
        buckets: List[list] = [[] for _ in range(P)]
        for ci, c in enumerate(classes):
            idx = rng.permutation(np.flatnonzero(y == c))
            cuts = (np.cumsum(props[ci])[:-1] * len(idx)).astype(int)
            for p, part in enumerate(np.split(idx, cuts)):
                buckets[p].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            break
    else:  # pragma: no cover
        raise RuntimeError("could not draw a partition with the requested minimum size")
    train_idx = [np.array(sorted(b), dtype=int) for b in buckets]
    tbuckets: List[list] = [[] for _ in range(P)]
    for ci, c in enumerate(classes):
        idx = np.flatnonzero(yte == c)
        owner = rng.choice(P, size=len(idx), p=props[ci])
        for i, p in zip(idx, owner):
            tbuckets[p].append(int(i))
    test_idx = [np.array(sorted(b), dtype=int) for b in tbuckets]
    return Partition(train_idx, test_idx, props)


def bootstrap_partition(y: np.ndarray, P: int, alpha: float, n_per_client: int,
                        rng: np.random.Generator) -> Partition:
    """Partition for the scalability study (P up to 500).

    Each client draws ``n_per_client`` training samples *with replacement*; its
    class mix follows Dir(alpha).  The population is therefore scaled without
    adding information, which is what a scheduling study needs.
    """
    classes = np.unique(y)
    pools = [np.flatnonzero(y == c) for c in classes]
    train_idx = []
    for _ in range(P):
        mix = rng.dirichlet(alpha * np.ones(len(classes)))
        counts = rng.multinomial(n_per_client, mix)
        idx = np.concatenate([rng.choice(pools[ci], size=n, replace=True)
                              for ci, n in enumerate(counts)]) if n_per_client else np.array([], int)
        train_idx.append(idx.astype(int))
    return Partition(train_idx, [np.array([], dtype=int) for _ in range(P)],
                     np.zeros((len(classes), P)))
