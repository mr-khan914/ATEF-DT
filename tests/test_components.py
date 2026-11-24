"""Unit tests of the building blocks: model, quantizer, trust layer, selection
rules, aggregation rules, energy model, twin and metrics."""
import itertools

import numpy as np
import pytest

from atefdt import aggregation as agg
from atefdt import selection as sel
from atefdt.config import Config, load_config
from atefdt.metrics import expected_jain_uniform, jain
from atefdt.model import MLP
from atefdt.quantization import (FULL_PRECISION_BITS, bitwidth, compression_factor, payload_bits, quantize,
                                 stability_surrogate)
from atefdt.system import System
from atefdt.trust import cosine, exclusion_bound, geometric_median, update_reputation
from atefdt.twin import KalmanTwin, SyntheticPatient, acuity, duty, fit_level, fit_transition


# ------------------------------------------------------------------ model
def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    m = MLP(5, 4)
    w = m.init(rng)
    X, y = rng.normal(size=(12, 5)), (rng.random(12) < 0.5).astype(float)
    _, g = m.loss_grad(w, X, y)
    for i in rng.choice(m.d, 10, replace=False):
        e = np.zeros(m.d)
        e[i] = 1e-6
        fd = (m.loss(w + e, X, y) - m.loss(w - e, X, y)) / 2e-6
        assert abs(fd - g[i]) < 1e-6


def test_parameter_count_matches_the_paper():
    assert MLP(30, 64).d == 2049


# ------------------------------------------------------------------ quantizer
def test_quantizer_is_unbiased_and_bounded():
    rng = np.random.default_rng(1)
    v = rng.normal(size=400)
    draws = np.stack([quantize(v, 4, rng) for _ in range(4000)])
    step = (v.max() - v.min()) / (2 ** 4 - 1)
    assert np.abs(draws.mean(axis=0) - v).max() < 0.05 * step * 4
    assert np.abs(draws - v).max() <= step + 1e-12
    assert len(np.unique(np.round((draws[0] - v.min()) / step))) <= 16


def test_bitwidth_rule_bounds_and_monotonicity():
    cfg = Config()
    h = np.array([0.05, 0.5, 1.0, 4.0])
    C = np.full(4, 100.0)
    for psi in (1.0, 0.3, 0.01):
        b = bitwidth(cfg, psi, h, 1.0, C)
        assert b.min() >= cfg.b_min and b.max() <= cfg.b_max
        assert np.all(np.diff(b) >= 0)                      # better channel, never fewer bits
    assert np.all(bitwidth(cfg, 1.0, h, 1.0, C) >= bitwidth(cfg, 0.1, h, 1.0, C))
    assert np.all(bitwidth(cfg, 1.0, h, 1.0, C, "full") == FULL_PRECISION_BITS)
    low = bitwidth(cfg, 1.0, np.array([4.0]), 1.0, np.array([cfg.C_ref / 4]))
    assert low[0] <= bitwidth(cfg, 1.0, np.array([4.0]), 1.0, np.array([cfg.C_ref]))[0]


def test_payload_and_compression_factor():
    cfg = Config()
    assert payload_bits(cfg, np.array([4]), 2049)[0] == 4 * 2049 + cfg.c_q
    assert payload_bits(cfg, np.array([32]), 2049)[0] == 32 * 2049
    assert compression_factor(cfg, 4.0, 2049) == pytest.approx(32 / (4 + 64 / 2049))


def test_stability_surrogate_starts_at_one():
    assert stability_surrogate(-1.0, -1.0) == 1.0
    assert stability_surrogate(0.5, 2.0) == 0.25


# ------------------------------------------------------------------ trust layer
def test_geometric_median_resists_an_outlier():
    rng = np.random.default_rng(2)
    U = rng.normal(0, 0.1, size=(9, 6)) + 1.0
    U[0] = 1e4
    z, it = geometric_median(U)
    assert np.linalg.norm(z - 1.0) < 0.5 and it >= 1
    assert np.linalg.norm(U.mean(axis=0) - 1.0) > 100


def test_reputation_stays_in_unit_interval_and_exclusion_bound():
    rho = np.array([1.0, 0.3, 0.0])
    for phi in (np.array([-1.0, 0.2, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, -1.0])):
        new = update_reputation(rho, phi, 0.6)
        assert np.all(new >= 0) and np.all(new <= 1)
    k = exclusion_bound(1.0, 0.05, 0.6)
    assert k == 6
    r = 1.0
    for i in range(1, k + 1):
        r = float(update_reputation(np.array([r]), np.array([-0.4]), 0.6)[0])
        assert (r < 0.05) == (i == k)                       # barred at exactly the k-th submission


def test_cosine_handles_zero_vectors():
    U = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
    assert np.allclose(cosine(U, np.array([2.0, 0.0])), [1.0, 0.0, -1.0])


# ------------------------------------------------------------------ selection rules
def _instance(seed, n=12):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 1.0, n), rng.integers(0, 9, n).astype(float), rng.uniform(0.05, 1.0, n)


def test_top_k_minimizes_the_log_domain_objective():
    """Lemma A.1: the k smallest scores minimize the sum of log-scores over all k-subsets."""
    cfg = Config()
    e, n, rho = _instance(3, 9)
    S = sel.select("product", cfg, np.ones(9, bool), 4, e, n, rho, np.random.default_rng(0), sel.init_state(9))
    logs = np.log(sel.product_score(cfg, *sel.factors(cfg, e, n, rho, np.arange(9))))
    best = min(itertools.combinations(range(9), 4), key=lambda c: logs[list(c)].sum())
    assert logs[S].sum() == pytest.approx(logs[list(best)].sum())


def test_product_rule_is_invariant_to_rescaling_and_the_weighted_sum_is_not():
    """Proposition 3."""
    cfg = Config()
    changed = 0
    for seed in range(40):
        e, n, rho = _instance(seed)
        feas, st, rng = np.ones(12, bool), sel.init_state(12), np.random.default_rng(0)
        base_p = sel.select("product", cfg, feas, 5, e, n, rho, rng, st)
        base_a = sel.select("additive", cfg, feas, 5, e, n, rho, rng, st)
        for c in (0.01, 0.1, 10.0, 1000.0):
            assert np.array_equal(base_p, sel.select("product", cfg, feas, 5, e, n, rho, rng, st, energy_scale=c))
            changed += not np.array_equal(base_a, sel.select("additive", cfg, feas, 5, e, n, rho, rng, st,
                                                             energy_scale=c))
    assert changed > 0


def test_selected_set_is_not_dominated():
    """No unselected feasible client is at least as good on all three factors and better on one."""
    cfg = Config()
    for seed in range(30):
        e, n, rho = _instance(seed)
        S = sel.select("product", cfg, np.ones(12, bool), 5, e, n, rho, np.random.default_rng(0), sel.init_state(12))
        f = np.stack(sel.factors(cfg, e, n, rho, np.arange(12)), axis=1)
        for q in set(range(12)) - set(S):
            for p in S:
                assert not (np.all(f[q] <= f[p]) and np.any(f[q] < f[p]))


def test_sequential_pipeline_can_miss_the_best_composite():
    """Proposition 2: an energy shortlist removes the client with the smallest composite score."""
    cfg = Config()
    e = np.array([0.10, 0.11, 0.12, 0.13])
    n = np.array([20.0, 20.0, 20.0, 0.0])
    rho = np.ones(4)
    feas, st, rng = np.ones(4, bool), sel.init_state(4), np.random.default_rng(0)
    assert list(sel.select("product", cfg, feas, 1, e, n, rho, rng, st)) == [3]
    assert 3 not in sel.select("seq_ef", cfg, feas, 1, e, n, rho, rng, st, shortlist=2)


def test_infeasible_clients_are_never_selected():
    cfg = Config()
    e, n, rho = _instance(5)
    feas = np.zeros(12, bool)
    feas[[1, 4, 7]] = True
    for rule in ("random", "energy", "product", "additive", "seq_ef", "seq_fe", "dpp", "fairfedcs", "all"):
        S = sel.select(rule, cfg, feas, 5, e, n, rho, np.random.default_rng(0), sel.init_state(12))
        assert set(S) <= {1, 4, 7} and len(S) == 3


# ------------------------------------------------------------------ aggregation
def test_reputation_coefficients_properties():
    rng = np.random.default_rng(6)
    for _ in range(50):
        k = int(rng.integers(2, 9))
        U = rng.normal(size=(k, 20)) * rng.uniform(0.1, 10, (k, 1))
        rho = rng.uniform(0.05, 1.0, k)
        a, acc = agg.reputation_coefficients(U, rho, 0.05)
        assert acc.all() and np.all(a >= 0)
        assert 0.05 / 2 - 1e-12 <= a.sum() <= 1 + 1e-12     # s in [rho_min / 2, 1] when nothing is rejected
        omega = a / agg.clip_factors(U)
        assert omega.sum() == pytest.approx(1.0)            # reputation weights sum to one
    a, acc = agg.reputation_coefficients(U, np.r_[0.01, rho[1:]], 0.05)
    assert not acc[0] and a[0] == 0.0


def test_clipping_bounds_every_update_by_the_median_norm():
    rng = np.random.default_rng(7)
    U = rng.normal(size=(7, 10)) * np.array([1, 1, 1, 1, 1, 50, 500])[:, None]
    c = agg.clip_factors(U)
    norms = np.linalg.norm(c[:, None] * U, axis=1)
    assert norms.max() <= np.median(np.linalg.norm(U, axis=1)) + 1e-9


def test_robust_baselines_ignore_a_gross_outlier():
    rng = np.random.default_rng(8)
    U = rng.normal(0, 0.05, size=(8, 12)) + 1.0
    U[3] = -1e3
    for f in (agg.cw_median, lambda u: agg.trimmed_mean(u, 0.2), agg.multi_krum,
              lambda u: agg.gm_aggregate(u, 1e-6, 200)):
        assert np.linalg.norm(f(U) - 1.0) < 0.5
    assert np.allclose(agg.fedavg_weights(np.array([1.0, 3.0])), [0.25, 0.75])


# ------------------------------------------------------------------ energy model
def test_energy_model_monotonicity():
    cfg = Config()
    sysm = System(cfg, 6, 4, np.random.default_rng(9))
    h = np.array([0.1, 1.0, 10.0])
    e = sysm.uplink_energy(1e5, h)
    assert np.all(np.diff(e) < 0)                           # better channel, less energy
    assert np.allclose(sysm.uplink_energy(2e5, h), 2 * e)   # linear in the payload
    assert np.all(sysm.downlink_energy(1e5, h) > 0)
    assert np.allclose(sysm.comp_energy(np.ones(6) * 10, 100), cfg.alpha * sysm.kappa * 10 * 100)
    assert sysm.available(np.array([1e-9]))[0] == False     # noqa: E712  (deep fade is an outage)


# ------------------------------------------------------------------ twin
def test_duty_and_acuity():
    assert acuity(np.zeros(3)) == 0.0 and acuity(np.array([0.0, 9.0, 1.0])) == 1.0
    assert duty(0.0, 0.2) == pytest.approx(0.2) and duty(1.0, 0.2) == pytest.approx(1.0)
    assert duty(0.3, 0.2) < duty(0.6, 0.2)


def test_level_twin_tracks_better_than_holding_the_last_measurement():
    err_twin, err_hold = [], []
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        pat = SyntheticPatient(3, np.random.default_rng(seed), episode=seed % 2 == 0, R=50, r_sd=0.4)
        win = pat.window(30)
        A, Q = fit_level(win, pat.r_sd ** 2)
        Rm = pat.r_sd ** 2 * np.eye(3)
        tw, last = KalmanTwin(A, Q, Rm, win[-1], Rm), win[-1]
        for _ in range(50):
            x = pat.step()
            y = pat.measure() if rng.random() < 0.8 else None
            tw.step(y)
            last = y if y is not None else last
            err_twin.append(np.abs(tw.u - x).mean())
            err_hold.append(np.abs(last - x).mean())
    assert np.mean(err_twin) < np.mean(err_hold)


def test_twin_identification_shapes_and_staleness():
    pat = SyntheticPatient(3, np.random.default_rng(0), episode=False, R=50)
    win = pat.window(30)
    for A, Q in (fit_level(win, 0.01), fit_transition(win, 1e-2)):
        assert A.shape == (3, 3) and Q.shape == (3, 3) and np.all(np.linalg.eigvalsh(Q) > 0)
    tw = KalmanTwin(np.eye(3), 0.01 * np.eye(3), 0.01 * np.eye(3), win[-1], np.eye(3))
    tw.step(None)
    tw.step(None)
    assert tw.stale == 2
    tw.step(win[-1])
    assert tw.stale == 0


# ------------------------------------------------------------------ metrics and configuration
def test_jain_index():
    assert jain([3, 3, 3, 3]) == pytest.approx(1.0)
    assert jain([5, 0, 0, 0]) == pytest.approx(0.25)
    assert 0.96 < expected_jain_uniform(20, 8, 50) < 0.98


def test_configuration_file_matches_the_paper():
    cfg = load_config()
    assert (cfg.P, cfg.S_max, cfg.R) == (20, 8, 50)
    assert (cfg.lam_E, cfg.lam_F, cfg.lam_T) == (1.0, 3.0, 0.4)
    assert (cfg.mu, cfg.rho_min) == (0.60, 0.05)
    with pytest.raises(AttributeError):
        cfg.copy(not_a_field=1)
