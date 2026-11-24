"""End-to-end tests of the simulator: determinism, accounting identities and the
agreement of a fresh run with the results shipped in results/."""
import json
from pathlib import Path

import numpy as np
import pytest

from atefdt.config import load_config
from atefdt.simulator import METHODS, run

CFG = load_config()
SHORT = CFG.copy(R=12)
ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_runs_and_accounts_consistently(method):
    r = run(SHORT, method, 0)
    assert 0.0 <= r["final_acc"] <= 1.0 and np.isfinite(r["final_loss"])
    assert r["E_total"] == pytest.approx(r["E_comp"] + r["E_ul"] + r["E_dl"] + r["E_sig"])
    assert r["E_cum"][-1] == pytest.approx(r["E_total"])
    assert r["n_tx"] == int(r["n_sel"].sum()) == int(np.sum(r["k"]))
    assert 0.0 < r["jain_all"] <= 1.0
    if METHODS[method].sel != "all":
        assert max(r["k"]) <= SHORT.S_max
    assert r["bits_down"] == 32 * r["d"] * r["n_tx"]


def test_runs_are_deterministic():
    a, b = run(SHORT, "atef_dt", 3), run(SHORT, "atef_dt", 3)
    assert a["final_loss"] == b["final_loss"] and a["E_total"] == b["E_total"]
    assert np.array_equal(a["n_sel"], b["n_sel"])
    assert run(SHORT, "atef_dt", 4)["final_loss"] != a["final_loss"]


def test_all_methods_share_partition_devices_and_fading():
    a, b = run(SHORT, "atef_dt", 1), run(SHORT, "fedavg_random", 1)
    assert np.array_equal(a["N"], b["N"])
    assert np.array_equal(a["kappa"], b["kappa"]) and np.array_equal(a["hbar"], b["hbar"])
    assert np.array_equal(a["n_avail"], b["n_avail"])


def test_uplink_volume_matches_the_bit_widths():
    r = run(SHORT, "atef_dt", 0, log_tx=True)
    rows = np.array(r["tx"])
    assert len(rows) == r["n_tx"]
    assert int((rows[:, 6] * r["d"]).sum()) == r["bits_payload"]
    assert r["bits_meta"] == SHORT.c_q * r["n_tx"]
    assert rows[:, 6].min() >= SHORT.b_min and rows[:, 6].max() <= SHORT.b_max
    assert run(SHORT, "ef_trust", 0)["bits_payload"] == 32 * r["d"] * r["n_tx"]


def test_reputation_gate_and_coefficients():
    r = run(SHORT, "atef_dt", 2, attack="label_flip", adv_frac=0.2)
    assert np.all(r["rho"] >= 0) and np.all(r["rho"] <= 1)
    active = r["s"][r["k"] > 0]
    assert np.all(active > 0) and np.all(active <= 1 + 1e-12)
    assert r["n_adv"] == 4 and 0 <= r["mal_rej"] <= r["mal_recv"]
    assert 0.0 <= r["mal_weight_share"] <= 100.0


def test_attacks_start_at_the_configured_round():
    clean = run(SHORT, "fedavg_random", 0)
    late = run(SHORT.copy(attack_start=12), "fedavg_random", 0, attack="scaling", adv_frac=0.2)
    assert np.allclose(clean["loss"], late["loss"])        # the attack never starts within 12 rounds
    early = run(SHORT.copy(attack_start=0), "fedavg_random", 0, attack="scaling", adv_frac=0.2)
    assert not np.allclose(clean["loss"], early["loss"])


def test_twin_stream_and_screens():
    c = SHORT.copy(battery_low=3.0, battery_high=7.0, C_res=1.0, C_ref=4.0)
    for scr in ("twin", "instant", "none"):
        r = run(c, "atef_dt", 0, stream=True, screen=scr)
        assert r["twin_mae"] > 0 and r["hold_mae"] > 0 and r["fl_breach"] >= 0


def test_scalability_path():
    r = run(SHORT, "atef_dt", 0, P=50, bootstrap=True)
    assert r["P"] == 50 and r["k_budget"] == 20 and max(r["k"]) <= 20


def test_fresh_run_agrees_with_the_shipped_results():
    """Guards the numbers of Table 5 against silent changes of the code."""
    path = ROOT / "results" / "main.json"
    if not path.exists():
        pytest.skip("results/main.json not present; run experiments/run_all.py first")
    stored = json.loads(path.read_text())["atef_dt"]
    fresh = [run(CFG, "atef_dt", s) for s in CFG.seeds]
    assert np.mean([r["final_acc"] for r in fresh]) == pytest.approx(np.mean(stored["final_acc"]), abs=0.015)
    assert np.mean([r["E_total"] for r in fresh]) == pytest.approx(np.mean(stored["E_total"]), rel=0.05)
    assert np.mean([r["jain_all"] for r in fresh]) == pytest.approx(np.mean(stored["jain_all"]), abs=0.02)
