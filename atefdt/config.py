"""Configuration: one dataclass holds every parameter of the simulation."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    # ---------------- experiment ----------------
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    # ---------------- data ----------------
    split_seed: int = 2026          # fixes the train/validation/test split for all runs
    test_frac: float = 0.20
    val_frac: float = 0.10          # fraction of the training split held out for validation
    dirichlet_alpha: float = 0.5    # label-skew concentration
    min_client_samples: int = 6
    # ---------------- federated learning ----------------
    P: int = 20                     # clients
    S_max: int = 8                  # participation budget per round
    R: int = 50                     # communication rounds
    lr: float = 1.0                 # local learning rate (eta)
    local_epochs: int = 1           # E (full-batch passes over the local data)
    hidden: int = 64
    # ---------------- devices ----------------
    kappa_low: float = 0.5
    kappa_high: float = 2.0
    battery_low: float = 80.0       # J
    battery_high: float = 200.0     # J
    hbar_mean: float = 1.0          # mean of the exponential law of the per-client mean gain
    # ---------------- energy / radio ----------------
    alpha: float = 1e-6             # J per parameter per sample per pass
    P_tx: float = 0.1               # W, radiated uplink power
    eta_pa: float = 0.35            # amplifier efficiency
    per: float = 1e-2               # packet error rate
    W: float = 1e6                  # Hz
    nu: float = 0.15                # header-and-coding overhead
    c_q: int = 64                   # quantizer metadata bits per payload
    noise_gap: float = 1e-3         # sigma_n^2 * W * Gamma (W)
    c_min: float = 1.0              # outage threshold vartheta_min (bit/s/Hz); below it: outage
    P_bs: float = 1.0               # W, downlink transmit power of the access point
    P_rx: float = 0.05              # W, receive-chain power of a device
    state_bits: int = 160           # five 32-bit scalars per state report
    # ---------------- quantization ----------------
    b_min: int = 4
    b_max: int = 16
    gamma_b: float = 0.5
    # ---------------- trust ----------------
    mu: float = 0.60
    rho_min: float = 0.05
    gm_tol: float = 1e-5
    gm_max_iter: int = 200
    ref_weighted: bool = False      # True: reputation-weighted reference (design alternative, Section 6.5)
    # ---------------- selection score ----------------
    lam_E: float = 1.0
    lam_F: float = 3.0
    lam_T: float = 0.4
    eps0: float = 0.1
    # ---------------- digital twin ----------------
    H: int = 5                      # forecast horizon (rounds)
    dt: float = 60.0                # s, duration of one round
    P_mon: float = 2e-3             # W, monitoring power at full duty
    duty_min: float = 0.2           # sensing duty at zero acuity
    C_res: float = 10.0             # J, battery reserve
    C_ref: float = 20.0             # J, battery level below which the payload is coarsened
    stale_max: int = 3              # staleness bound (rounds without a measurement)
    twin_m: int = 3                 # physiological variables per patient
    twin_fit_window: int = 30       # samples used to identify A_p
    twin_ridge: float = 1e-2
    twin_obs_prob: float = 0.8      # probability that a measurement arrives in a round
    twin_model: str = "level"       # level (A = I) | ar (ridge least squares)
    twin_r_sd: float = 0.25         # sensor-noise standard deviation of the synthetic stream
    # ---------------- attacks ----------------
    attack_start: int = 10
    # ---------------- baselines ----------------
    ffcs_sigma: float = 0.6         # FairFedCS trade-off weight (value of the original paper)
    dpp_V: float = 1.0              # drift-plus-penalty weight
    trim_beta: float = 0.2
    seq_shortlist: int = 16         # 2 * S_max

    def copy(self, **changes: Any) -> "Config":
        c = copy.deepcopy(self)
        for k, v in changes.items():
            if not hasattr(c, k):
                raise AttributeError(f"unknown configuration field: {k}")
            setattr(c, k, v)
        return c

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None) -> Config:
    """Read a YAML file and overlay it on the defaults."""
    cfg = Config()
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"unknown configuration field in {path}: {k}")
            if isinstance(getattr(cfg, k), float) and v is not None:
                v = float(v)
            setattr(cfg, k, v)
    return cfg
