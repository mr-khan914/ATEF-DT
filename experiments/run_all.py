#!/usr/bin/env python3
"""Run every experiment of the paper and store the results as JSON/CSV.

    python experiments/run_all.py            # full protocol (five seeds)
    python experiments/run_all.py --quick    # two seeds, reduced grids (smoke test)
    python experiments/run_all.py --only main byzantine

Outputs: results/<experiment>.json and results/logs/tx_<method>.csv
(one row per client-round transmission).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atefdt.attacks import ATTACKS  # noqa: E402
from atefdt.config import load_config  # noqa: E402
from atefdt.metrics import dump, expected_jain_uniform, paired_p  # noqa: E402
from atefdt.simulator import METHODS, get_split, run  # noqa: E402

RES = ROOT / "results"
MAIN = ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "energy_trust", "energy_fair", "atef_dt"]
SCALARS = ["final_acc", "final_loss", "val_loss", "E_total", "E_comp", "E_ul", "E_dl", "E_sig",
           "jain_all", "jain_acc", "jain_honest", "n_tx", "bits_payload", "bits_meta", "bits_down",
           "bits_state", "mean_b_run", "sensitivity", "specificity", "s_mean", "s_min"]
OPTIONAL = ["acc_high", "acc_low", "sel_share_high", "n_test_high", "n_test_low", "pi_l1", "c_part",
            "c_bound", "zeta", "rej_rate", "mal_recv", "mal_rej", "n_adv", "n_adv_barred",
            "honest_false_excl", "viol_share", "mean_adv_share", "mal_weight_share", "mal_sel_share",
            "asr", "below_reserve", "n_below", "min_battery", "twin_mae", "hold_mae", "fl_breach", "depleted"]
CURVES = ["loss", "acc", "jain", "E_cum", "mean_b", "psi", "rho_honest", "rho_byz", "s", "n_avail"]
TX_HEADER = ["seed", "round", "client", "N_p", "kappa_p", "h", "bits_per_coord", "E_comp_J", "E_uplink_J",
             "E_downlink_J", "accepted", "malicious", "coefficient"]


def many(cfg, method, seeds, **kw):
    return [run(cfg, method, s, **kw) for s in seeds]


def summ(runs, curves=False):
    out = {k: [r[k] for r in runs] for k in SCALARS}
    for k in OPTIONAL:
        if all(k in r for r in runs):
            out[k] = [r[k] for r in runs]
    if curves:
        for k in CURVES:
            out["curve_" + k] = np.nanmean(np.array([r[k] for r in runs], float), axis=0)
    return out


# ------------------------------------------------------------------ experiments
def exp_setup(cfg, seeds, quick):
    """Data statistics, learning-rate selection on the validation split, outage rate."""
    d = get_split(cfg)
    out = {"n_train": len(d.ytr), "n_val": len(d.yval), "n_test": len(d.yte), "lr_grid": {}}
    for lr in ([0.5, 1.0] if quick else [0.1, 0.2, 0.5, 1.0]):
        rs = many(cfg.copy(lr=lr), "fedavg_full", seeds)
        out["lr_grid"][str(lr)] = {"val_loss": [r["val_loss"] for r in rs], "val_acc": [r["val_acc"] for r in rs]}
    rs = many(cfg, "fedavg_full", seeds)
    out["np_min"] = int(min(r["N"].min() for r in rs))
    out["np_max"] = int(max(r["N"].max() for r in rs))
    out["outage_pct"] = float(100 * (1 - np.mean([r["n_avail"].mean() for r in rs]) / cfg.P))
    out["d"] = int(rs[0]["d"])
    out["jain_uniform_expected"] = expected_jain_uniform(cfg.P, cfg.S_max, cfg.R)
    means = {k: float(np.mean(v["val_loss"])) for k, v in out["lr_grid"].items()}
    out["lr_selected"] = float(min(means, key=means.get))
    return out


def exp_main(cfg, seeds, quick):
    """Tables 5 and 6: all methods without attack; per-transmission logs are written to CSV."""
    out = {}
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    for mk in MAIN + ["ef_trust"]:
        rs = many(cfg, mk, seeds, log_tx=True, checkpoints=True)
        out[mk] = summ(rs, curves=True)
        with open(RES / "logs" / f"tx_{mk}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(TX_HEADER)
            for r in rs:
                wr.writerows(r["tx"])
    ref = out["fedavg_full"]["final_acc"]
    for mk in out:
        out[mk]["p_vs_full"] = paired_p(out[mk]["final_acc"], ref)
        out[mk]["p_vs_atef"] = paired_p(out[mk]["final_acc"], out["atef_dt"]["final_acc"])
    return out


def exp_quant(cfg, seeds, quick):
    """Bit-width policies with the selection rule held fixed (Table 7)."""
    return {mk: summ(many(cfg, mk, seeds), curves=True) for mk in ["atef_q16", "atef_q8", "atef_qglobal", "atef_dt"]}


def exp_fairness(cfg, seeds, quick):
    """Sweep of the fairness exponent: Jain index, accuracy, participation bias and its bound."""
    grid = [0.0, 3.0] if quick else [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    return {str(lf): summ(many(cfg, "atef_dt", seeds, lam=(cfg.lam_E, lf, cfg.lam_T), checkpoints=True))
            for lf in grid}


def exp_schedulers(cfg, seeds, quick):
    """Energy-fairness frontiers of five scheduling rules, matched comparison, rescaling test."""
    out = {"frontier": {}, "rescale": {}}
    g = (lambda full, q: q if quick else full)
    fam = {
        "product": [("atef_dt", dict(lam=(cfg.lam_E, v, cfg.lam_T)), v) for v in g([0, .5, 1, 1.5, 2, 2.5, 3, 4, 6], [0, 3])],
        "additive": [("wsum", dict(lam=(cfg.lam_E, v, cfg.lam_T)), v) for v in g([0, .25, .5, .75, .9, 1, 1.5, 2, 3], [0, 1])],
        "dpp": [("dpp", dict(), v) for v in g([0.5, 2, 5, 7.5, 10, 15, 20, 40, 80, 160], [5, 40])],
        "seq_ef": [("seq_ef", dict(shortlist=v), v) for v in g([8, 10, 12, 14, 16, 17, 18, 19, 20], [8, 20])],
        "seq_fe": [("seq_fe", dict(shortlist=v), v) for v in g([8, 9, 10, 12, 14, 16, 18, 20], [8, 20])],
    }
    for name, items in fam.items():
        out["frontier"][name] = []
        for mk, kw, v in items:
            c = cfg.copy(dpp_V=float(v)) if name == "dpp" else cfg
            s = summ(many(c, mk, seeds, **kw))
            s["knob"] = v
            out["frontier"][name].append(s)
    out["frontier"]["fairfedcs"] = [dict(summ(many(cfg, "fairfedcs", seeds)), knob=cfg.ffcs_sigma)]
    # rescaling of the energy factor: the product ranking is invariant, the additive one is not
    for name, mk in (("product", "atef_dt"), ("additive", "wsum")):
        out["rescale"][name] = {}
        for sc in [0.1, 1.0, 10.0]:
            rs = many(cfg, mk, seeds, energy_scale=sc)
            s = summ(rs)
            s["n_sel"] = [r["n_sel"] for r in rs]
            out["rescale"][name][str(sc)] = s
    return out


def exp_byzantine(cfg, seeds, quick):
    """Attack grid, 20 % detail, comparison across methods, reputation dynamics."""
    out = {"methods": {}, "grid": {}, "curves": {}}
    for atk_name in ["sign_flip", "label_flip"]:
        out["methods"][atk_name] = {
            mk: summ(many(cfg, mk, seeds, attack=atk_name, adv_frac=0.2), curves=True)
            for mk in ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "energy_fair", "energy_trust",
                       "ef_trust", "atef_dt",
                       "atef_norep", "atef_norep_gm", "seq_ef", "seq_fe", "wsum", "dpp",
                       "ra_cwmed", "ra_trimmed", "ra_mkrum", "ra_gm"]}
    for mk in ["atef_norep", "atef_norep_gm", "seq_ef", "seq_fe", "wsum", "dpp",
               "ra_cwmed", "ra_trimmed", "ra_mkrum", "ra_gm"]:
        out["methods"].setdefault("clean", {})[mk] = summ(many(cfg, mk, seeds))
    fracs = [0.2] if quick else [0.1, 0.2, 0.3, 0.4]
    for a in (["sign_flip", "backdoor"] if quick else ATTACKS):
        out["grid"][a] = {}
        for fr in fracs:
            rs = many(cfg, "atef_dt", seeds, attack=a, adv_frac=fr)
            s = summ(rs, curves=(fr == 0.2))
            s["subs_to_bar"] = [float(x) for r in rs for x in np.atleast_1d(r["subs_to_bar"])]
            out["grid"][a][str(fr)] = s
    # stress comparison beyond the domain of Assumption 1
    out["high"] = {}
    for atk_name in ["sign_flip", "label_flip"]:
        for fr in ([0.4] if quick else [0.3, 0.4]):
            out["high"][f"{atk_name}|{fr:g}"] = {
                mk: summ(many(cfg, mk, seeds, attack=atk_name, adv_frac=fr))
                for mk in ["atef_dt", "atef_norep", "ra_gm", "ra_mkrum", "fedavg_random"]}
    out["backdoor"] = {mk: summ(many(cfg, mk, seeds, attack="backdoor", adv_frac=0.2))
                       for mk in ["fedavg_random", "atef_norep", "ra_gm", "ra_mkrum"]}
    out["backdoor_clean"] = {mk: summ(many(cfg, mk, seeds)) for mk in ["fedavg_random", "atef_dt"]}
    return out


def exp_trust(cfg, seeds, quick):
    """Sensitivity of the reputation layer: forgetting factor, threshold, attack onset and
    the reputation-weighted reference that was examined and not adopted."""
    out = {"grid": {}, "onset": {}, "weighted_ref": {}}
    mus = [0.6] if quick else [0.5, 0.6, 0.7, 0.8]
    rmins = [0.05] if quick else [0.02, 0.05, 0.10]
    for mu in mus:
        for rm in rmins:
            c = cfg.copy(mu=mu, rho_min=rm)
            out["grid"][f"{mu:g}|{rm:g}"] = {
                "clean": summ(many(c, "atef_dt", seeds)),
                "label_flip": summ(many(c, "atef_dt", seeds, attack="label_flip", adv_frac=0.2)),
                "sign_flip": summ(many(c, "atef_dt", seeds, attack="sign_flip", adv_frac=0.2))}
    for onset in [0, 10]:
        c = cfg.copy(attack_start=onset)
        out["onset"][str(onset)] = {a: summ(many(c, "atef_dt", seeds, attack=a, adv_frac=0.2))
                                    for a in ["sign_flip", "label_flip"]}
    c = cfg.copy(ref_weighted=True)
    out["weighted_ref"] = {"clean": summ(many(c, "atef_dt", seeds)),
                           "label_flip": summ(many(c, "atef_dt", seeds, attack="label_flip", adv_frac=0.2))}
    return out


def exp_twin(cfg, seeds, quick):
    """Functional test of the twin layer on synthetic physiological streams (Section 6.5, Table 17).

    Batteries are made small enough for the reserve to bind.  Three feasibility
    screens are compared, and the state-tracking error of two twin models is set
    against holding the last received measurement.
    """
    small = dict(battery_low=3.0, battery_high=7.0, C_res=1.0, C_ref=4.0)
    c = cfg.copy(**small)
    out = {"config": dict(small, twin_r_sd=cfg.twin_r_sd, twin_obs_prob=cfg.twin_obs_prob, H=cfg.H)}
    for scr in ["twin", "instant", "none"]:
        out[scr] = summ(many(c, "atef_dt", seeds, stream=True, screen=scr))
    out["ar"] = summ(many(c.copy(twin_model="ar"), "atef_dt", seeds, stream=True, screen="twin"))
    out["noise"] = {}
    for sd in ([0.25] if quick else [0.10, 0.25, 0.40]):
        out["noise"][f"{sd:g}"] = {mdl: summ(many(c.copy(twin_model=mdl, twin_r_sd=sd), "atef_dt", seeds,
                                                  stream=True, screen="twin")) for mdl in ["level", "ar"]}
    return out


def exp_scalability(cfg, seeds, quick):
    """Population from 20 to 500 clients (bootstrap partition) and server-side overhead."""
    out = {}
    for P in ([20, 100] if quick else [20, 50, 100, 200, 500]):
        out[str(P)] = {}
        for mk in ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "atef_dt"]:
            rs = many(cfg, mk, seeds, P=P, bootstrap=True)
            s = summ(rs)
            s["timing"] = {k: float(np.mean([r["timing"][k] for r in rs])) for k in rs[0]["timing"]}
            out[str(P)][mk] = s
    return out


def exp_sensitivity(cfg, seeds, quick):
    """Sensitivity of the energy comparison to alpha and to the amplifier efficiency."""
    out = {"alpha": {}, "eta_pa": {}}
    meth = ["fedavg_full", "fedavg_random", "ea_fl", "atef_dt"]
    for a in ([1e-8, 1e-6] if quick else [1e-9, 1e-8, 1e-7, 1e-6]):
        out["alpha"][f"{a:g}"] = {mk: summ(many(cfg.copy(alpha=a), mk, seeds)) for mk in meth}
    for e in ([0.35] if quick else [0.2, 0.35, 0.5]):
        out["eta_pa"][f"{e:g}"] = {mk: summ(many(cfg.copy(eta_pa=e), mk, seeds)) for mk in meth}
    return out


EXPERIMENTS = {"setup": exp_setup, "main": exp_main, "quant": exp_quant, "fairness": exp_fairness,
               "schedulers": exp_schedulers, "byzantine": exp_byzantine, "trust": exp_trust, "twin": exp_twin,
               "scalability": exp_scalability, "sensitivity": exp_sensitivity}


def main():
    global RES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML configuration (default: configs/default.yaml)")
    ap.add_argument("--quick", action="store_true", help="two seeds and reduced grids")
    ap.add_argument("--only", nargs="*", default=None, choices=list(EXPERIMENTS))
    ap.add_argument("--out", default=str(RES))
    args = ap.parse_args()
    RES = Path(args.out)
    cfg = load_config(args.config)
    seeds = cfg.seeds[:2] if args.quick else cfg.seeds
    RES.mkdir(parents=True, exist_ok=True)
    dump({"config": cfg.to_dict(), "seeds": seeds, "quick": args.quick,
          "methods": {k: v.label for k, v in METHODS.items()}}, RES / "meta.json")
    for name in (args.only or list(EXPERIMENTS)):
        t0 = time.time()
        dump(EXPERIMENTS[name](cfg, seeds, args.quick), RES / f"{name}.json")
        print(f"[{name:12s}] done in {time.time() - t0:6.1f} s", flush=True)


if __name__ == "__main__":
    main()
