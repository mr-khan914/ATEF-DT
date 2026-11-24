"""Round-level simulator: one function, ``run``, executes a complete training run
of one method under one seed and returns every quantity reported in the paper."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import aggregation as agg
from . import attacks as atk
from . import selection as sel
from .config import Config
from .data import Split, bootstrap_partition, dirichlet_partition, load_split
from .metrics import jain
from .model import MLP
from .quantization import (FULL_PRECISION_BITS, bitwidth, payload_bits, quantize,
                           stability_surrogate)
from .system import System
from .trust import cosine, geometric_median, update_reputation
from .twin import KalmanTwin, SyntheticPatient, acuity, duty, fit_level, fit_transition


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    sel: str                    # all | random | energy | fairfedcs | product | additive | seq_ef | seq_fe | dpp
    agg: str                    # fedavg | atef | clipmean | clipgm | cwmed | trimmed | mkrum | gm
    quant: str = "full"         # full | fixed16 | fixed8 | global | perclient
    trust: bool = False         # reputation layer: admission, gate and weights
    screen: str = "none"        # twin | instant | none (battery feasibility screen)
    no_trust_factor: bool = False   # lambda_T = 0 in the score
    uses_state: bool = False    # the method needs the pre-selection state report


_A = dict(agg="atef", quant="perclient", trust=True, screen="twin", uses_state=True)
METHODS: Dict[str, Method] = {m.key: m for m in [
    # ---- reference methods and baselines (Section 5.1) ----
    Method("fedavg_full", "FedAvg-Full", "all", "fedavg"),
    Method("fedavg_random", "FedAvg-Random", "random", "fedavg"),
    Method("ea_fl", "EA-FL", "energy", "fedavg", screen="instant", uses_state=True),
    Method("fairfedcs", "FairFedCS", "fairfedcs", "fedavg"),
    Method("energy_trust", "Energy+Trust", "energy", "atef", trust=True, screen="instant", uses_state=True),
    Method("energy_fair", "Energy+Fair", "product", "fedavg", screen="twin", no_trust_factor=True, uses_state=True),
    Method("ef_trust", "Energy+Fair+Trust", "product", "atef", trust=True, screen="twin", uses_state=True),
    Method("atef_dt", "ATEF-DT", "product", **_A),
    # ---- bit-width policies with the selection rule held fixed ----
    Method("atef_q16", "Fixed 16-bit", "product", **{**_A, "quant": "fixed16"}),
    Method("atef_q8", "Fixed 8-bit", "product", **{**_A, "quant": "fixed8"}),
    Method("atef_qglobal", "Global schedule", "product", **{**_A, "quant": "global"}),
    # ---- multi-objective schedulers, everything but the selection rule identical ----
    Method("wsum", "Weighted sum", "additive", **_A),
    Method("dpp", "Drift-plus-penalty", "dpp", **_A),
    Method("seq_ef", "Seq-EF", "seq_ef", **_A),
    Method("seq_fe", "Seq-FE", "seq_fe", **_A),
    # ---- matched variants without the reputation layer ----
    Method("atef_norep", "ATEF-DT-Rep", "product", "clipmean", quant="perclient", screen="twin",
           no_trust_factor=True, uses_state=True),
    Method("atef_norep_gm", "ATEF-DT-Rep+GM", "product", "clipgm", quant="perclient", screen="twin",
           no_trust_factor=True, uses_state=True),
    # ---- robust-aggregation baselines: uniform selection, full precision ----
    Method("ra_cwmed", "Coordinate-wise median", "random", "cwmed"),
    Method("ra_trimmed", "Trimmed mean", "random", "trimmed"),
    Method("ra_mkrum", "Multi-Krum", "random", "mkrum"),
    Method("ra_gm", "Geometric median", "random", "gm"),
]}

_SPLIT_CACHE: Dict[tuple, Split] = {}


def get_split(cfg: Config) -> Split:
    key = (cfg.split_seed, cfg.test_frac, cfg.val_frac)
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = load_split(cfg)
    return _SPLIT_CACHE[key]


def run(cfg: Config, method: str, seed: int, attack: Optional[str] = None, adv_frac: float = 0.0,
        P: Optional[int] = None, bootstrap: bool = False, stream: bool = False,
        screen: Optional[str] = None, lam=None, energy_scale: float = 1.0,
        shortlist: Optional[int] = None, log_tx: bool = False, checkpoints: bool = False) -> Dict:
    """Run ``method`` for ``cfg.R`` rounds with the random streams of ``seed``.

    Partition, model initialization, device parameters and the whole fading
    sequence depend on ``seed`` only, so they are identical for every method.
    """
    m = METHODS[method]
    screen = screen or m.screen
    P = P or cfg.P
    k_budget = cfg.S_max if P == cfg.P else int(round(P * cfg.S_max / cfg.P))
    R, E = cfg.R, cfg.local_epochs
    ss = np.random.SeedSequence([int(seed), 20261115])
    r_part, r_init, r_sys, r_q, r_sel, r_adv, r_str = [np.random.default_rng(s) for s in ss.spawn(7)]

    data = get_split(cfg)
    if bootstrap:
        n_mean = int(round(len(data.ytr) / cfg.P))
        part = bootstrap_partition(data.ytr, P, cfg.dirichlet_alpha, n_mean, r_part)
    else:
        part = dirichlet_partition(data.ytr, data.yte, P, cfg.dirichlet_alpha, cfg.min_client_samples, r_part)
    Xp = [data.Xtr[i] for i in part.train_idx]
    yp = [data.ytr[i] for i in part.train_idx]
    N = part.sizes.astype(float)
    n_frac = N / N.sum()

    model = MLP(data.Xtr.shape[1], cfg.hidden)
    d = model.d
    w = model.init(r_init)
    system = System(cfg, P, R, r_sys)
    h_ref = float(system.hbar.mean())
    C = system.C0.copy()

    # ---------------- adversaries ----------------
    n_adv = int(round(adv_frac * P)) if attack else 0
    adv = np.zeros(P, bool)
    if n_adv:
        adv[r_adv.choice(P, n_adv, replace=False)] = True
    poisoned = {}
    if attack == "backdoor":
        for p in np.flatnonzero(adv):
            poisoned[p] = atk.poison_backdoor(Xp[p], yp[p], r_adv)
    harmful_dir = r_adv.normal(size=d)

    # ---------------- digital twin ----------------
    mon_unit = cfg.P_mon * cfg.dt                       # J per round at full duty
    twins, patients = [], []
    if stream:
        for p in range(P):
            pat = SyntheticPatient(cfg.twin_m, np.random.default_rng(r_str.integers(1 << 31)),
                                   episode=bool(r_str.random() < 0.5), R=R, r_sd=cfg.twin_r_sd)
            win = pat.window(cfg.twin_fit_window)
            if cfg.twin_model == "level":
                A, Q = fit_level(win, pat.r_sd ** 2)
            else:
                A, Q = fit_transition(win, cfg.twin_ridge)
            Rm = (pat.r_sd ** 2) * np.eye(cfg.twin_m)
            twins.append(KalmanTwin(A, Q, Rm, win[-1], Rm))
            patients.append(pat)

    # ---------------- state ----------------
    rho = np.ones(P)
    n_sel = np.zeros(P)
    n_acc = np.zeros(P)
    state = sel.init_state(P)
    norm_first = norm_last = -1.0
    lam_use = lam
    if m.no_trust_factor and lam is None:
        lam_use = (cfg.lam_E, cfg.lam_F, 0.0)

    out = {k: [] for k in ["loss", "acc", "jain", "e_comp", "e_ul", "e_dl", "e_sig", "k", "mean_b",
                           "s", "psi", "rho_honest", "rho_byz", "adv_share", "n_avail", "n_feas"]}
    bits_payload = bits_meta = bits_down = bits_state = 0
    n_tx = 0
    b_all = []
    a_sum = np.zeros(P)
    s_sum = 0.0
    mal_recv = mal_rej = 0
    mal_subs = np.zeros(P)
    first_rej = np.full(P, -1.0)
    ever_barred = np.zeros(P, bool)
    below_reserve = 0
    ever_below = np.zeros(P, bool)
    twin_err, hold_err = [], []
    y_last = [None] * P
    last_part = np.full(P, -10 ** 6)       # round of the latest participation
    part_flagged = np.zeros(P, bool)       # latest participation already counted as a breach
    fl_breach = 0                          # participations followed by a reserve breach within H rounds
    depleted = 0                           # client-rounds with an empty battery
    tx_rows = []
    ckpt = []
    timing = {"score": [], "gm": [], "rep": [], "agg": [], "iters": []}
    viol_rounds = attack_rounds = 0
    mal_w = tot_w = 0.0            # aggregation weight carried by malicious updates
    mal_sel = tot_sel = 0          # selections that went to adversaries after onset

    for r in range(R):
        h = system.h[r]
        # ---- twin: advance, monitoring drain now and forecast over H rounds ----
        if stream:
            mon = np.empty(P)
            mon_fc = np.empty(P)
            for p in range(P):
                x = patients[p].step()
                y = patients[p].measure() if r_str.random() < cfg.twin_obs_prob else None
                twins[p].step(y)
                if y is not None:
                    y_last[p] = y
                twin_err.append(float(np.mean(np.abs(twins[p].u - x))))
                if y_last[p] is not None:
                    hold_err.append(float(np.mean(np.abs(y_last[p] - x))))
                mon[p] = mon_unit * duty(acuity(x), cfg.duty_min)
                if twins[p].stale > cfg.stale_max:          # stale twin: conservative forecast
                    mon_fc[p] = cfg.H * mon_unit
                else:
                    mon_fc[p] = mon_unit * sum(duty(acuity(twins[p].predict(j)), cfg.duty_min)
                                               for j in range(cfg.H))
        else:
            mon = np.full(P, mon_unit * cfg.duty_min)
            mon_fc = cfg.H * mon

        # ---- pre-selection quantities (state report, Eqs. (9)-(10)) ----
        t0 = time.perf_counter()
        avail = system.available(h) & (C > 0)
        psi = stability_surrogate(norm_last, norm_first)
        b = bitwidth(cfg, psi, h, h_ref, C, m.quant)
        bits = payload_bits(cfg, b, d)
        e_comp = system.comp_energy(N, d, E)
        e_ul = system.uplink_energy(bits, h)
        e_dl = system.downlink_energy(FULL_PRECISION_BITS * d, h)
        e_hat = e_comp + e_ul + e_dl
        if screen == "twin":
            batt_ok = (C - e_hat - mon_fc) >= cfg.C_res
        elif screen == "instant":
            batt_ok = (C - e_hat) >= cfg.C_res
        else:
            batt_ok = (C - e_hat) >= 0.0
        feas = avail & batt_ok
        if m.trust:
            feas &= rho >= cfg.rho_min
        S = sel.select(m.sel, cfg, feas, k_budget, e_hat, n_sel, rho, r_sel, state,
                       lam=lam_use, energy_scale=energy_scale, shortlist=shortlist)
        t_score = time.perf_counter() - t0
        n_sel[S] += 1
        k = len(S)

        # ---- energy and battery ----
        e_sig_vec = np.zeros(P)
        if m.uses_state:                                   # state report of every reachable device
            e_sig_vec[avail] = system.uplink_energy(cfg.state_bits, h[avail])
        e_sig = float(e_sig_vec.sum())
        out["e_comp"].append(float(e_comp[S].sum()))
        out["e_ul"].append(float(e_ul[S].sum()))
        out["e_dl"].append(float(e_dl[S].sum()))
        out["e_sig"].append(e_sig)
        C[S] -= e_hat[S]
        C -= mon + e_sig_vec
        C = np.maximum(C, 0.0)
        last_part[S] = r
        part_flagged[S] = False
        low = C < cfg.C_res
        below_reserve += int(low.sum())
        ever_below |= low
        depleted += int((C <= 0).sum())
        hit = low & ((r - last_part) <= cfg.H) & ~part_flagged
        fl_breach += int(hit.sum())
        part_flagged |= hit
        if m.uses_state:
            bits_state += cfg.state_bits * int(avail.sum())

        a_round = np.zeros(P)
        accepted = np.ones(k, bool)
        s_r = 0.0
        if k > 0:
            # ---- local training (Eq. (20)): E full-batch passes ----
            U = np.empty((k, d))
            honest_rows = []
            is_mal = np.zeros(k, bool)
            attacking = bool(attack) and r >= cfg.attack_start
            for j, p in enumerate(S):
                X, y = Xp[p], yp[p]
                mal = attacking and adv[p]
                is_mal[j] = mal
                if mal and attack == "label_flip":
                    y = 1.0 - y
                elif mal and attack == "backdoor":
                    X, y = poisoned[p]
                wl = w.copy()
                for _ in range(E):
                    wl -= cfg.lr * model.loss_grad(wl, X, y)[1]
                dw = wl - w
                if mal and attack in ("sign_flip", "scaling", "gaussian"):
                    dw = atk.simple_attack(attack, dw, r_adv)
                elif mal and attack == "backdoor":
                    dw = atk.BACKDOOR_BOOST * dw
                U[j] = dw
                if not mal:
                    honest_rows.append(j)
            if attacking and attack == "adaptive" and is_mal.any():
                ref = U[honest_rows] if honest_rows else U
                U[is_mal] = atk.adaptive_update(ref, harmful_dir)
            # ---- quantization of the transmitted vectors ----
            if m.quant != "full":
                for j, p in enumerate(S):
                    U[j] = quantize(U[j], int(b[p]), r_q)
            n_tx += k
            b_all.extend(b[S].tolist())
            bits_payload += int((b[S] * d).sum())
            bits_meta += int((bits[S] - b[S] * d).sum())
            bits_down += FULL_PRECISION_BITS * d * k
            if attacking:
                attack_rounds += 1
                share = float(is_mal.mean())
                out["adv_share"].append(share)
                viol_rounds += int(share >= 0.5)
                mal_sel += int(is_mal.sum())
                tot_sel += k

            # ---- aggregation ----
            t_gm = t_rep = 0.0
            t1 = time.perf_counter()
            if m.agg == "fedavg":
                a = agg.fedavg_weights(N[S])
                step = a @ U
                s_r = 1.0
                if m.sel == "fairfedcs":      # leave-one-out contribution on the validation split
                    base = model.loss(w + step, data.Xval, data.yval)
                    contrib = np.empty(k)
                    for j in range(k):
                        keep = np.arange(k) != j
                        aj = agg.fedavg_weights(N[S][keep]) if keep.any() else np.array([])
                        wj = w + (aj @ U[keep] if keep.any() else 0.0)
                        contrib[j] = model.loss(wj, data.Xval, data.yval) - base
                    sel.update_state("fairfedcs", cfg, state, S, k_budget, contrib)
            elif m.agg == "atef":
                tg = time.perf_counter()
                ref, iters = geometric_median(U, cfg.gm_tol, cfg.gm_max_iter,
                                              weights=rho[S] if cfg.ref_weighted else None)
                t_gm = time.perf_counter() - tg
                tr = time.perf_counter()
                phi = cosine(U, ref)
                rho_new = update_reputation(rho[S], phi, cfg.mu)
                t_rep = time.perf_counter() - tr
                a, accepted = agg.reputation_coefficients(U, rho_new, cfg.rho_min)
                step = a @ U
                s_r = float(a.sum())
                rho[S] = rho_new
                timing["iters"].append(iters)
                if attacking:
                    for j, p in enumerate(S):
                        if is_mal[j]:
                            mal_recv += 1
                            mal_subs[p] += 1
                            if not accepted[j]:
                                mal_rej += 1
                                if first_rej[p] < 0:
                                    first_rej[p] = mal_subs[p]
                ever_barred |= rho < cfg.rho_min
            elif m.agg == "clipmean":
                a = agg.clip_factors(U) / k
                step = a @ U
                s_r = float(a.sum())
            elif m.agg == "clipgm":
                a = np.zeros(k)
                step = agg.gm_aggregate(agg.clip_factors(U)[:, None] * U, cfg.gm_tol, cfg.gm_max_iter)
            elif m.agg == "cwmed":
                a, step = np.zeros(k), agg.cw_median(U)
            elif m.agg == "trimmed":
                a, step = np.zeros(k), agg.trimmed_mean(U, cfg.trim_beta)
            elif m.agg == "mkrum":
                a, step = np.zeros(k), agg.multi_krum(U)
            elif m.agg == "gm":
                a, step = np.zeros(k), agg.gm_aggregate(U, cfg.gm_tol, cfg.gm_max_iter)
            else:
                raise ValueError(m.agg)
            t_agg = time.perf_counter() - t1 - t_gm - t_rep
            w = w + step
            if attacking and a.sum() > 0:
                mal_w += float(a[is_mal].sum())
                tot_w += float(a.sum())
            n_acc[S[accepted]] += 1
            a_round[S] = a
            a_sum += a_round
            s_sum += s_r
            norm_last = float(np.linalg.norm(step))
            if norm_first < 0 and norm_last > 0:
                norm_first = norm_last
            timing["score"].append(t_score * 1e3)
            timing["gm"].append(t_gm * 1e3)
            timing["rep"].append(t_rep * 1e3)
            timing["agg"].append(t_agg * 1e3)
            if log_tx:
                for j, p in enumerate(S):
                    tx_rows.append((seed, r, int(p), int(N[p]), float(system.kappa[p]), float(h[p]),
                                    int(b[p]), float(e_comp[p]), float(e_ul[p]), float(e_dl[p]),
                                    int(accepted[j]), int(is_mal[j]), float(a[j])))
        if m.sel == "dpp":
            sel.update_state("dpp", cfg, state, S, k_budget)

        # ---- per-round log ----
        out["loss"].append(model.loss(w, data.Xte, data.yte))
        out["acc"].append(model.accuracy(w, data.Xte, data.yte))
        out["jain"].append(jain(n_sel))
        out["k"].append(k)
        out["mean_b"].append(float(b[S].mean()) if k else float("nan"))
        out["psi"].append(psi)
        out["s"].append(s_r)
        out["rho_honest"].append(float(rho[~adv].mean()))
        out["rho_byz"].append(float(rho[adv].mean()) if adv.any() else float("nan"))
        out["n_avail"].append(int(avail.sum()))
        out["n_feas"].append(int(feas.sum()))
        if checkpoints and (r + 1) % 10 == 0:
            G = np.stack([model.loss_grad(w, Xp[p], yp[p])[1] for p in range(P)])
            ckpt.append(G)

    # ---------------- end-of-run summaries ----------------
    res = {k_: np.array(v) for k_, v in out.items()}
    res.update(method=method, seed=seed, P=P, k_budget=k_budget, d=d,
               N=N, kappa=system.kappa, hbar=system.hbar, n_sel=n_sel, n_acc=n_acc,
               final_acc=float(res["acc"][-1]), final_loss=float(res["loss"][-1]),
               val_loss=model.loss(w, data.Xval, data.yval), val_acc=model.accuracy(w, data.Xval, data.yval),
               E_comp=float(res["e_comp"].sum()), E_ul=float(res["e_ul"].sum()),
               E_dl=float(res["e_dl"].sum()), E_sig=float(res["e_sig"].sum()),
               n_tx=n_tx, bits_payload=bits_payload, bits_meta=bits_meta, bits_down=bits_down,
               bits_state=bits_state, mean_b_run=float(np.mean(b_all)) if b_all else float("nan"),
               jain_all=jain(n_sel), jain_acc=jain(n_acc),
               jain_honest=jain(n_sel[~adv]), s_mean=s_sum / max(sum(res["k"] > 0), 1),
               s_min=float(res["s"][res["k"] > 0].min()) if (res["k"] > 0).any() else float("nan"),
               below_reserve=below_reserve, n_below=int(ever_below.sum()),
               fl_breach=fl_breach, depleted=depleted,
               min_battery=float(C.min()), adv=adv, rho=rho)
    res["E_total"] = res["E_comp"] + res["E_ul"] + res["E_dl"] + res["E_sig"]
    res["E_cum"] = np.cumsum(res["e_comp"] + res["e_ul"] + res["e_dl"] + res["e_sig"])
    # sensitivity / specificity on the test split
    pred = model.logits(w, data.Xte) > 0
    res["sensitivity"] = float(pred[data.yte == 1].mean())
    res["specificity"] = float((~pred[data.yte == 0]).mean())
    # group-level evaluation by device-resource tier (static cost at the mean channel gain)
    if not bootstrap:
        cost = system.comp_energy(N, d, E) + system.uplink_energy(FULL_PRECISION_BITS * d, system.hbar)
        high = cost > np.median(cost)
        for name, mask in (("high", high), ("low", ~high)):
            idx = np.concatenate([part.test_idx[p] for p in np.flatnonzero(mask)]).astype(int)
            res[f"acc_{name}"] = model.accuracy(w, data.Xte[idx], data.yte[idx])
            res[f"n_test_{name}"] = int(len(idx))
            res[f"sel_share_{name}"] = float(n_sel[mask].sum() / max(n_sel.sum(), 1))
    # effective aggregation weights and participation bias (Section 4.1)
    if s_sum > 0:
        pi_hat = a_sum / a_sum.sum()
        res["pi_hat"] = pi_hat
        res["pi_l1"] = float(np.abs(pi_hat - n_frac).sum())
        if ckpt:
            cp, zt = [], []
            for G in ckpt:
                gF = n_frac @ G
                cp.append(float(np.linalg.norm((pi_hat - n_frac) @ G)))
                zt.append(float(np.linalg.norm(G - gF, axis=1).max()))
            res["c_part"] = float(np.mean(cp))
            res["zeta"] = float(np.mean(zt))
            res["c_bound"] = float(np.mean(np.array(zt) * res["pi_l1"]))
    # trust statistics
    if m.trust:
        barred_adv = adv & ever_barred
        res.update(mal_recv=mal_recv, mal_rej=mal_rej,
                   rej_rate=100.0 * mal_rej / mal_recv if mal_recv else float("nan"),
                   n_adv=int(adv.sum()), n_adv_barred=int(barred_adv.sum()),
                   subs_to_bar=first_rej[barred_adv & (first_rej > 0)],
                   honest_false_excl=100.0 * float((ever_barred & ~adv).sum()) / max(int((~adv).sum()), 1))
    if attack:
        res["viol_share"] = 100.0 * viol_rounds / max(attack_rounds, 1)
        res["mean_adv_share"] = 100.0 * float(np.mean(res["adv_share"])) if len(res["adv_share"]) else float("nan")
        res["mal_weight_share"] = 100.0 * mal_w / tot_w if tot_w > 0 else float("nan")
        res["mal_sel_share"] = 100.0 * mal_sel / tot_sel if tot_sel > 0 else float("nan")
    if attack == "backdoor" or attack is None:
        Xt = atk.add_trigger(data.Xte[data.yte == 1])
        res["asr"] = 100.0 * float((model.logits(w, Xt) <= 0).mean())
    if stream:
        res["twin_mae"] = float(np.mean(twin_err))
        res["hold_mae"] = float(np.mean(hold_err))
    res["timing"] = {k_: (float(np.mean(v)) if v else 0.0) for k_, v in timing.items()}
    if log_tx:
        res["tx"] = tx_rows
    return res
