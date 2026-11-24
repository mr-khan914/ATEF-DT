"""Patient-centric digital twin (Section 3.2): Kalman state tracking, acuity,
monitoring duty and the residual-energy forecast that feeds the feasibility screen.

The WDBC benchmark has no physiological stream; there the twin reduces to a
constant monitoring duty.  ``SyntheticPatient`` provides a clearly synthetic
linear-Gaussian stream that is used only by the twin-layer functional test.

Two transition models are available: the local-level model (A = I, default)
and a first-order autoregression identified by ridge least squares.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fit_transition(Y: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    """Identify A_p by ridge least squares on a monitoring window Y (T x m).

    Returns (A, Q) with Q the covariance of the one-step residuals.
    """
    Y0, Y1 = Y[:-1], Y[1:]
    m = Y.shape[1]
    A = (Y1.T @ Y0) @ np.linalg.inv(Y0.T @ Y0 + ridge * np.eye(m))
    res = Y1 - Y0 @ A.T
    Q = (res.T @ res) / max(len(res) - 1, 1) + 1e-6 * np.eye(m)
    return A, Q


def fit_level(Y: np.ndarray, r_var: float) -> Tuple[np.ndarray, np.ndarray]:
    """Local-level (random-walk-plus-noise) twin: A = I.

    The process-noise variances follow from the method of moments on the first
    differences of the window, Var(y_t - y_{t-1}) = Q + 2 R, floored at 1e-3.
    """
    m = Y.shape[1]
    q = np.maximum(np.var(np.diff(Y, axis=0), axis=0, ddof=1) - 2.0 * r_var, 1e-3)
    return np.eye(m), np.diag(q)


class KalmanTwin:
    """Predictor-corrector twin, Eqs. (6a)-(6d)."""

    def __init__(self, A, Q, Rm, u0, P0):
        self.A, self.Q, self.Rm = np.asarray(A), np.asarray(Q), np.asarray(Rm)
        self.u, self.Pc = np.asarray(u0, float).copy(), np.asarray(P0, float).copy()
        self.m = len(self.u)
        self.stale = 0

    def step(self, y: Optional[np.ndarray]) -> None:
        u_pred = self.A @ self.u
        P_pred = self.A @ self.Pc @ self.A.T + self.Q
        if y is None:                       # no measurement: open-loop propagation
            self.u, self.Pc = u_pred, P_pred
            self.stale += 1
            return
        K = P_pred @ np.linalg.inv(P_pred + self.Rm)          # H = I
        self.u = u_pred + K @ (y - u_pred)
        self.Pc = (np.eye(self.m) - K) @ P_pred
        self.stale = 0

    def predict(self, j: int) -> np.ndarray:
        """j-step open-loop prediction A^j u."""
        return np.linalg.matrix_power(self.A, j) @ self.u


def acuity(u: np.ndarray) -> float:
    """Largest standardized deviation over the m channels, saturating at three SD."""
    return float(min(1.0, np.max(np.abs(u)) / 3.0))


def duty(a: float, duty_min: float) -> float:
    """Monitoring duty l(a) = l_min + (1 - l_min) a, non-decreasing in acuity."""
    return duty_min + (1.0 - duty_min) * a


class SyntheticPatient:
    """Synthetic vital-sign deviations (standardized units): mean-reverting AR(1)
    dynamics with an optional deterioration episode that ramps the mean upward."""

    def __init__(self, m: int, rng: np.random.Generator, episode: bool, R: int, r_sd: float = 0.25):
        self.rng = rng
        self.A = np.diag(rng.uniform(0.80, 0.92, m))
        self.q_sd = 0.20
        self.r_sd = r_sd
        self.x = rng.normal(0.0, 0.3, m)
        self.t = 0
        self.onset = int(rng.integers(8, max(9, R - 15))) if episode else None
        self.peak = rng.uniform(2.5, 4.0, m) * rng.choice([1.0, 0.6, 0.3], m, replace=False) if episode else None

    def _mean(self) -> np.ndarray:
        if self.onset is None or self.t < self.onset:
            return np.zeros(len(self.x))
        return self.peak * min(1.0, (self.t - self.onset + 1) / 8.0)

    def step(self) -> np.ndarray:
        m = self._mean()
        self.x = m + self.A @ (self.x - m) + self.rng.normal(0.0, self.q_sd, len(self.x))
        self.t += 1
        return self.x

    def measure(self) -> np.ndarray:
        return self.x + self.rng.normal(0.0, self.r_sd, len(self.x))

    def window(self, T: int) -> np.ndarray:
        """Pre-run monitoring window used to identify the twin (stationary phase).

        The clock is reset afterwards, so a deterioration episode can only start
        during the training run and never inside the identification window.
        """
        Y = np.array([(self.step(), self.measure())[1] for _ in range(T)])
        self.t = 0
        return Y
