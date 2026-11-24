"""Attack models of Section 6.5.  Adversaries act from round ``attack_start`` on."""
from __future__ import annotations

import numpy as np

ATTACKS = ["sign_flip", "gaussian", "label_flip", "scaling", "backdoor", "adaptive"]

LABELS = {
    "sign_flip": "Sign flipping",
    "gaussian": "Gaussian poisoning",
    "label_flip": "Label flipping",
    "scaling": "Scaled / boosted",
    "backdoor": "Backdoor insertion",
    "adaptive": "Adaptive (angular)",
}

STRENGTH = {
    "sign_flip": r"update multiplied by $-1$",
    "gaussian": r"update replaced by a draw from $\mathcal{N}(\mathbf{0},(3\sigma_u)^2\mathbf{I})$, with $\sigma_u$ the root mean square of the update the client would have sent",
    "label_flip": r"local labels inverted before training",
    "scaling": r"update multiplied by $10$",
    "backdoor": r"trigger of four standard deviations on three features, target class benign, half of the local samples poisoned, update multiplied by $5$",
    "adaptive": r"identical colluding updates at cosine $0.1$ to the mean of the honest updates, along a common random direction, with the median honest norm",
}

SCALE_FACTOR = 10.0
BACKDOOR_BOOST = 5.0
TRIGGER_FEATURES = (0, 1, 2)
TRIGGER_VALUE = 4.0
ADAPTIVE_COSINE = 0.1
GAUSS_MULT = 3.0


def add_trigger(X: np.ndarray) -> np.ndarray:
    Xt = X.copy()
    Xt[:, list(TRIGGER_FEATURES)] = TRIGGER_VALUE
    return Xt


def poison_backdoor(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Half of the local samples carry the trigger and the target label (benign = 0)."""
    n = len(y)
    pick = rng.permutation(n)[: max(1, n // 2)]
    Xp, yp = X.copy(), y.copy()
    Xp[pick] = add_trigger(X[pick])
    yp[pick] = 0.0
    return Xp, yp


def simple_attack(name: str, dw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if name == "sign_flip":
        return -dw
    if name == "scaling":
        return SCALE_FACTOR * dw
    if name == "gaussian":
        return rng.normal(0.0, GAUSS_MULT * np.sqrt(np.mean(dw ** 2)) + 1e-12, dw.shape)
    raise ValueError(name)


def adaptive_update(honest: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Angular-evasion adversary: stays inside the acceptance region of a cosine test.

    ``honest`` holds the honest updates of the round (omniscient colluders);
    ``direction`` is the common harmful direction shared by the colluders.
    """
    g = honest.mean(axis=0)
    g_hat = g / max(np.linalg.norm(g), 1e-300)
    z = direction - (direction @ g_hat) * g_hat
    z /= max(np.linalg.norm(z), 1e-300)
    norm = np.median(np.linalg.norm(honest, axis=1))
    return norm * (ADAPTIVE_COSINE * g_hat + np.sqrt(1.0 - ADAPTIVE_COSINE ** 2) * z)
