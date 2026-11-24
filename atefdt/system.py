"""Devices, wireless channel and energy model (Section 3.3 of the manuscript)."""
from __future__ import annotations

import numpy as np

from .config import Config


class System:
    """Heterogeneous devices with Rayleigh block fading.

    All random quantities are drawn once in the constructor from ``rng``; the
    whole fading sequence is therefore replayed unchanged for every method
    that is run with the same seed.
    """

    def __init__(self, cfg: Config, P: int, R: int, rng: np.random.Generator):
        self.cfg = cfg
        self.kappa = rng.uniform(cfg.kappa_low, cfg.kappa_high, P)
        self.hbar = rng.exponential(cfg.hbar_mean, P)            # mean gain per client
        self.C0 = rng.uniform(cfg.battery_low, cfg.battery_high, P)
        self.h = self.hbar[None, :] * rng.exponential(1.0, (R, P))  # h_p^(r)

    # ---------------- radio ----------------
    def spectral_eff(self, h):
        """log2(1 + P_tx h / (sigma^2 W Gamma)) in bit/s/Hz (uplink)."""
        return np.log2(1.0 + self.cfg.P_tx * np.asarray(h) / self.cfg.noise_gap)

    def available(self, h):
        """A device is in outage when its uplink spectral efficiency is below c_min."""
        return self.spectral_eff(h) >= self.cfg.c_min

    def uplink_energy(self, bits, h):
        """Eqs. (12)-(13): radiated power x airtime, inflated by retransmissions and PA loss."""
        c = self.cfg
        airtime = (1.0 + c.nu) * np.asarray(bits, float) / (c.W * self.spectral_eff(h))
        return c.P_tx / (c.eta_pa * (1.0 - c.per)) * airtime

    def downlink_energy(self, bits, h):
        """Receive-chain energy while the global model is broadcast to the device."""
        c = self.cfg
        rate = c.W * np.log2(1.0 + c.P_bs * np.asarray(h) / c.noise_gap)
        return c.P_rx * (1.0 + c.nu) * np.asarray(bits, float) / rate

    # ---------------- computation ----------------
    def comp_energy(self, N, d: int, E: int = 1):
        """Eq. (11): alpha * kappa_p * N_p * d * E."""
        return self.cfg.alpha * self.kappa * np.asarray(N, float) * d * E
