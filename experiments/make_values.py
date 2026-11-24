#!/usr/bin/env python3
"""Turn results/*.json into the numbers and table bodies used by the manuscript.

    python experiments/make_values.py

Writes paper/values.tex (one \\defval{key}{value} per number quoted in the text)
and paper/tables/*.tex (table bodies).  The manuscript contains no hand-typed
result: every figure in it is produced here from the logs of run_all.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atefdt import attacks as atk  # noqa: E402
from atefdt.metrics import load  # noqa: E402
from atefdt.quantization import FULL_PRECISION_BITS  # noqa: E402
from atefdt.trust import exclusion_bound  # noqa: E402

SHORT = {"fedavg_full": "full", "fedavg_random": "rand", "ea_fl": "eafl", "fairfedcs": "ffcs",
         "energy_trust": "etrust", "energy_fair": "efair", "ef_trust": "eftrust", "atef_dt": "atef"}
LABEL = {"fedavg_full": "FedAvg-Full", "fedavg_random": "FedAvg-Random", "ea_fl": "EA-FL",
         "fairfedcs": "FairFedCS", "energy_trust": "Energy+Trust", "energy_fair": "Energy+Fair",
         "ef_trust": "Energy+Fair+Trust", "atef_dt": r"\textbf{ATEF-DT}"}
MAIN = ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "energy_trust", "energy_fair", "atef_dt"]
ASHORT = {"sign_flip": "sf", "gaussian": "ga", "label_flip": "lf", "scaling": "sc", "backdoor": "bd",
          "adaptive": "ad"}
PM = r"\,$\pm$\,"
VAL = {}


def mean(x):
    return float(np.nanmean(np.asarray(x, float)))


def sd(x):
    x = np.asarray(x, float)
    return float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0


def put(key, value, nd=None):
    VAL[key] = f"{value:.{nd}f}" if nd is not None else str(value)


def pm(x, nd, scale=1.0):
    x = np.asarray(x, float) * scale
    return f"{mean(x):.{nd}f}{PM}{sd(x):.{nd}f}"


def write_table(name, rows, out):
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / f"{name}.tex").write_text("\n".join(rows) + "\n")


def mb(bits):
    return mean(bits) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper"))
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, setup, M = load(res / "meta.json"), load(res / "setup.json"), load(res / "main.json")
    Q, F, S = load(res / "quant.json"), load(res / "fairness.json"), load(res / "schedulers.json")
    B, T, TW = load(res / "byzantine.json"), load(res / "trust.json"), load(res / "twin.json")
    SC, SE = load(res / "scalability.json"), load(res / "sensitivity.json")
    cfg = meta["config"]
    d = setup["d"]

    # ------------------------------------------------------------ setup
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    put("nseeds", words.get(len(meta["seeds"]), str(len(meta["seeds"]))))
    for k in ("n_train", "n_val", "n_test", "np_min", "np_max", "d"):
        put(k.replace("_", ""), setup[k])
    put("outage", setup["outage_pct"], 1)
    put("jainexp", setup["jain_uniform_expected"], 3)
    put("lr", setup["lr_selected"], 1)
    for lr, v in setup["lr_grid"].items():
        put("lrval-" + lr.replace(".", ""), mean(v["val_loss"]), 4)
    put("kstar", exclusion_bound(1.0, cfg["rho_min"], cfg["mu"]))

    # ------------------------------------------------------------ main comparison
    for mk in MAIN + ["ef_trust"]:
        v, s = M[mk], SHORT[mk]
        put(f"acc-{s}", 100 * mean(v["final_acc"]), 2)
        put(f"accsd-{s}", 100 * sd(v["final_acc"]), 2)
        put(f"sens-{s}", 100 * mean(v["sensitivity"]), 1)
        put(f"spec-{s}", 100 * mean(v["specificity"]), 1)
        put(f"E-{s}", mean(v["E_total"]), 2)
        put(f"Esd-{s}", sd(v["E_total"]), 2)
        put(f"Ecomp-{s}", mean(v["E_comp"]), 2)
        put(f"Eul-{s}", mean(v["E_ul"]), 2)
        put(f"Edl-{s}", mean(v["E_dl"]), 2)
        put(f"Esig-{s}", mean(v["E_sig"]), 3)
        put(f"jain-{s}", mean(v["jain_all"]), 3)
        put(f"jainsd-{s}", sd(v["jain_all"]), 3)
        put(f"tx-{s}", mean(v["n_tx"]), 0)
        put(f"upMb-{s}", mb(np.add(v["bits_payload"], v["bits_meta"])), 1)
        put(f"downMb-{s}", mb(v["bits_down"]), 1)
        put(f"p-{s}", v["p_vs_full"], 2)
        put(f"selhigh-{s}", 100 * mean(v["sel_share_high"]), 1)
        put(f"acchigh-{s}", 100 * mean(v["acc_high"]), 1)
        put(f"acclow-{s}", 100 * mean(v["acc_low"]), 1)
        put(f"pil1-{s}", mean(v["pi_l1"]), 2)
        put(f"cpart-{s}", mean(v["c_part"]), 3)
        put(f"cbound-{s}", mean(v["c_bound"]), 3)
    a, fu, ra, ef, eft = (M[k] for k in ("atef_dt", "fedavg_full", "fedavg_random", "energy_fair", "ef_trust"))
    put("sav-full", 100 * (1 - mean(a["E_total"]) / mean(fu["E_total"])), 1)
    put("sav-rand", 100 * (1 - mean(a["E_total"]) / mean(ra["E_total"])), 1)
    put("sav-efair", 100 * (1 - mean(a["E_total"]) / mean(ef["E_total"])), 1)
    put("sav-rand-full", 100 * (1 - mean(ra["E_total"]) / mean(fu["E_total"])), 1)
    put("sav-eafl-full", 100 * (1 - mean(M["ea_fl"]["E_total"]) / mean(fu["E_total"])), 1)
    put("extra-eafl", 100 * (mean(a["E_total"]) / mean(M["ea_fl"]["E_total"]) - 1), 1)
    put("accgap-full", 100 * (mean(fu["final_acc"]) - mean(a["final_acc"])), 2)
    put("accgap-rand", 100 * (mean(ra["final_acc"]) - mean(a["final_acc"])), 2)
    put("sensgap-full", 100 * (mean(fu["sensitivity"]) - mean(a["sensitivity"])), 1)
    put("compshare-full", 100 * mean(fu["E_comp"]) / mean(fu["E_total"]), 1)
    put("compshare-atef", 100 * mean(a["E_comp"]) / mean(a["E_total"]), 1)
    put("Ecomp-sav-rand", mean(ra["E_comp"]) - mean(a["E_comp"]), 2)
    put("Eul-sav-rand", mean(ra["E_ul"]) - mean(a["E_ul"]), 2)
    put("ul-red-eftrust", 100 * (1 - mean(a["E_ul"]) / mean(eft["E_ul"])), 1)
    put("meanb", mean(a["mean_b_run"]), 2)
    put("smean", mean(a["s_mean"]), 2)
    put("smin", float(np.min(a["s_min"])), 2)
    put("fe-clean", mean(a["honest_false_excl"]), 1)
    put("payloadMb", mb(a["bits_payload"]), 2)
    put("metaMb", mb(a["bits_meta"]), 3)
    put("stateMb", mb(a["bits_state"]), 3)
    up_a = mb(np.add(a["bits_payload"], a["bits_meta"]))
    up_f = mb(np.add(eft["bits_payload"], eft["bits_meta"]))
    put("upMb-atef2", up_a, 2)
    put("compr", up_f / up_a, 1)
    put("compr-formula", FULL_PRECISION_BITS / (mean(a["mean_b_run"]) + cfg["c_q"] / d), 1)
    tot_a = up_a + mb(a["bits_down"]) + mb(a["bits_state"])
    tot_f = up_f + mb(eft["bits_down"]) + mb(eft["bits_state"])
    put("totMb-atef", tot_a, 1)
    put("totMb-eftrust", tot_f, 1)
    put("tot-red", 100 * (1 - tot_a / tot_f), 1)
    put("up-red", 100 * (1 - up_a / up_f), 1)
    from scipy import stats as _st0
    put("p-sens-atef", float(_st0.ttest_rel(a["sensitivity"], fu["sensitivity"]).pvalue), 2)
    put("p-E-atef-rand", float(_st0.ttest_rel(a["E_total"], ra["E_total"]).pvalue), 3)
    put("p-acc-atef-rand", float(_st0.ttest_rel(a["final_acc"], ra["final_acc"]).pvalue), 2)
    put("loss-atef", mean(a["final_loss"]), 3)
    put("loss-full", mean(fu["final_loss"]), 3)
    put("lambdaF", cfg["lam_F"], 1)
    put("below-main", int(sum(np.sum(M[mk]["n_below"]) for mk in MAIN)))

    rows = []
    for mk in MAIN:
        v = M[mk]
        p = "--" if mk == "fedavg_full" else f"{v['p_vs_full']:.2f}"
        rows.append(f"{LABEL[mk]} & {pm(v['final_acc'], 2, 100)} & {pm(v['sensitivity'], 1, 100)} & "
                    f"{pm(v['specificity'], 1, 100)} & {pm(v['E_total'], 2)} & {pm(v['jain_all'], 3)} & "
                    f"{mean(v['n_tx']):.0f} & {p} \\\\")
    write_table("tab_main", rows, out)

    rows = []
    for mk in MAIN:
        v = M[mk]
        e = mean(v["E_total"])
        rows.append(f"{LABEL[mk]} & {mean(v['E_comp']):.2f} & {mean(v['E_ul']):.2f} & {mean(v['E_dl']):.2f} & "
                    f"{mean(v['E_sig']):.3f} & {e:.2f} & {100 * (1 - e / mean(fu['E_total'])):.1f} & "
                    f"{100 * (1 - e / mean(ra['E_total'])):+.1f} \\\\".replace("+", "$+$").replace("& -", "& $-$"))
    write_table("tab_energy", rows, out)

    rows = []
    for mk in MAIN:
        v = M[mk]
        gap = 100 * (mean(v["acc_high"]) - mean(v["acc_low"]))
        rows.append(f"{LABEL[mk]} & {100 * mean(v['sel_share_high']):.1f} & {pm(v['acc_high'], 1, 100)} & "
                    f"{pm(v['acc_low'], 1, 100)} & {gap:+.1f} & {mean(v['pi_l1']):.2f} \\\\"
                    .replace("+", "$+$").replace("& -", "& $-$"))
    write_table("tab_group", rows, out)

    # ------------------------------------------------------------ bit-width policies and traffic
    rows = []
    pol = [("Full precision (32 bit)", eft), ("Fixed 16 bit", Q["atef_q16"]), ("Fixed 8 bit", Q["atef_q8"]),
           ("Global schedule", Q["atef_qglobal"]), (r"\textbf{Per-client (ATEF-DT)}", Q["atef_dt"])]
    for name, v in pol:
        up = mb(np.add(v["bits_payload"], v["bits_meta"]))
        rows.append(f"{name} & {mean(v['mean_b_run']):.2f} & {mb(v['bits_payload']):.2f} & "
                    f"{mb(v['bits_meta']):.3f} & {up:.2f} & {up_f / up:.1f} & {mean(v['E_ul']):.3f} & "
                    f"{pm(v['final_acc'], 2, 100)} \\\\")
    write_table("tab_volume", rows, out)
    for key, v in (("q16", Q["atef_q16"]), ("q8", Q["atef_q8"]), ("qglobal", Q["atef_qglobal"])):
        put(f"acc-{key}", 100 * mean(v["final_acc"]), 2)
        put(f"meanb-{key}", mean(v["mean_b_run"]), 2)
        put(f"Eul-{key}", mean(v["E_ul"]), 3)

    # ------------------------------------------------------------ fairness exponent
    for lf, v in F.items():
        k = lf.replace(".", "")
        put(f"jain-lf{k}", mean(v["jain_all"]), 3)
        put(f"acc-lf{k}", 100 * mean(v["final_acc"]), 2)
        put(f"E-lf{k}", mean(v["E_total"]), 2)
        put(f"cpart-lf{k}", mean(v["c_part"]), 3)
        put(f"cbound-lf{k}", mean(v["c_bound"]), 3)
        put(f"pil1-lf{k}", mean(v["pi_l1"]), 2)

    # ------------------------------------------------------------ schedulers
    FAM = [("product", r"\textbf{Weighted product (ATEF-DT)}", r"$\lambda_F$"),
           ("additive", "Weighted sum", r"$\lambda_F$"),
           ("dpp", "Drift-plus-penalty", r"$V$"),
           ("seq_ef", "Sequential, energy then fairness", r"$\bar k$"),
           ("seq_fe", "Sequential, fairness then energy", r"$\bar k$")]
    target = 0.95
    put("jain-req", target, 2)
    rows = []
    for fam, name, knob in FAM:
        ok = [it for it in S["frontier"][fam] if mean(it["jain_all"]) >= target]
        it = min(ok, key=lambda z: mean(z["E_total"]))
        resc = "--"
        if fam in S["rescale"]:
            base = np.asarray(S["rescale"][fam]["1.0"]["n_sel"], float)
            ch = {sc: 100 * np.abs(np.asarray(v["n_sel"], float) - base).sum() / base.sum()
                  for sc, v in S["rescale"][fam].items()}
            resc = f"{ch['0.1']:.1f} / {ch['10.0']:.1f}"
            short = "prod" if fam == "product" else "add"
            put(f"resc-{short}-01", ch["0.1"], 1)
            put(f"resc-{short}-10", ch["10.0"], 1)
            put(f"rescjain-{short}-10", mean(S["rescale"][fam]["10.0"]["jain_all"]), 3)
            put(f"rescjain-{short}-1", mean(S["rescale"][fam]["1.0"]["jain_all"]), 3)
            put(f"rescE-{short}-10", mean(S["rescale"][fam]["10.0"]["E_total"]), 2)
            put(f"rescE-{short}-1", mean(S["rescale"][fam]["1.0"]["E_total"]), 2)
        kn = f"{it['knob']:g}"
        rows.append(f"{name} & {knob}$\\,=\\,${kn} & {pm(it['jain_all'], 3)} & {pm(it['E_total'], 2)} & "
                    f"{pm(it['final_acc'], 2, 100)} & {resc} \\\\")
        short = {"product": "prod", "additive": "add", "dpp": "dpp", "seq_ef": "seqef", "seq_fe": "seqfe"}[fam]
        put(f"sched-E-{short}", mean(it["E_total"]), 2)
        put(f"sched-Esd-{short}", sd(it["E_total"]), 2)
        put(f"sched-J-{short}", mean(it["jain_all"]), 3)
        put(f"sched-knob-{short}", kn)
    from scipy import stats as _st
    chosen = {}
    for fam, _, _ in FAM:
        ok = [z for z in S["frontier"][fam] if mean(z["jain_all"]) >= target]
        chosen[fam] = min(ok, key=lambda z: mean(z["E_total"]))
    for fam, short in (("additive", "add"), ("dpp", "dpp"), ("seq_ef", "seqef"), ("seq_fe", "seqfe")):
        diff = np.asarray(chosen[fam]["E_total"]) - np.asarray(chosen["product"]["E_total"])
        put(f"sched-diff-{short}", float(diff.mean()), 2)
        put(f"sched-p-{short}", float(_st.ttest_rel(chosen[fam]["E_total"], chosen["product"]["E_total"]).pvalue), 3)
    it = S["frontier"]["fairfedcs"][0]
    rows.append(f"FairFedCS~\\cite{{shi2023fairfedcs}} & $\\sigma\\,=\\,${it['knob']:g} & {pm(it['jain_all'], 3)} & "
                f"{pm(it['E_total'], 2)} & {pm(it['final_acc'], 2, 100)} & -- \\\\")
    write_table("tab_sched", rows, out)
    put("sched-gap-add", mean(M["atef_dt"]["E_total"]) - VAL_float("sched-E-add"), 2)

    # ------------------------------------------------------------ Byzantine
    BM = ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "energy_fair", "energy_trust", "atef_dt"]
    rows = []
    for mk in BM:
        sfv, lfv = B["methods"]["sign_flip"][mk], B["methods"]["label_flip"][mk]
        rows.append(f"{LABEL[mk]} & {pm(M[mk]['final_acc'], 1, 100)} & {pm(sfv['final_acc'], 1, 100)} & "
                    f"{pm(lfv['final_acc'], 1, 100)} \\\\")
        put(f"sf20-{SHORT[mk]}", 100 * mean(sfv["final_acc"]), 1)
        put(f"lf20-{SHORT[mk]}", 100 * mean(lfv["final_acc"]), 1)
        put(f"lf20sd-{SHORT[mk]}", 100 * sd(lfv["final_acc"]), 1)
    write_table("tab_byz", rows, out)
    _lf = B["methods"]["label_flip"]
    put("p-lf-atef-rand", float(_st0.ttest_rel(_lf["atef_dt"]["final_acc"], _lf["fedavg_random"]["final_acc"]).pvalue), 2)
    put("p-lf-atef-norep", float(_st0.ttest_rel(_lf["atef_dt"]["final_acc"], _lf["atef_norep"]["final_acc"]).pvalue), 2)

    fr_list = sorted(next(iter(B["grid"].values())).keys(), key=float)
    rows, rows_r = [], []
    for at_name in atk.ATTACKS:
        if at_name not in B["grid"]:
            continue
        g = B["grid"][at_name]
        rows.append(f"{atk.LABELS[at_name]} & " + " & ".join(pm(g[fr]["final_acc"], 1, 100) for fr in fr_list) + r" \\")
        for fr in fr_list:
            k = f"{ASHORT[at_name]}{int(round(float(fr) * 100))}"
            v = g[fr]
            put(f"g-acc-{k}", 100 * mean(v["final_acc"]), 1)
            put(f"g-accsd-{k}", 100 * sd(v["final_acc"]), 1)
            put(f"g-malsel-{k}", mean(v["mal_sel_share"]), 1)
            put(f"g-malw-{k}", mean(v["mal_weight_share"]), 1)
            put(f"g-rej-{k}", mean(v["rej_rate"]), 1)
            put(f"g-bar-{k}", int(np.sum(v["n_adv_barred"])))
            put(f"g-nadv-{k}", int(np.sum(v["n_adv"])))
            put(f"g-fe-{k}", mean(v["honest_false_excl"]), 1)
            put(f"g-viol-{k}", mean(v["viol_share"]), 1)
            if at_name == "backdoor":
                put(f"g-asr-{k}", mean(v["asr"]), 0)
    write_table("tab_grid", rows, out)
    for fr in fr_list:
        share = mean([mean(B["grid"][a_][fr]["mean_adv_share"]) for a_ in B["grid"]])
        viol = mean([mean(B["grid"][a_][fr]["viol_share"]) for a_ in B["grid"]])
        rows_r.append(f"{int(round(float(fr) * 100))} & {share:.1f} & {viol:.1f} \\\\")
        put(f"viol-{int(round(float(fr) * 100))}", viol, 1)
    write_table("tab_realized", rows_r, out)

    rows = []
    for at_name in atk.ATTACKS:
        if at_name not in B["grid"] or "0.2" not in B["grid"][at_name]:
            continue
        v = B["grid"][at_name]["0.2"]
        asr = f"{mean(v['asr']):.0f}" if at_name == "backdoor" else "--"
        rows.append(f"{atk.LABELS[at_name]} & {pm(v['final_acc'], 1, 100)} & {mean(v['mal_sel_share']):.1f} & "
                    f"{mean(v['mal_weight_share']):.1f} & {mean(v['rej_rate']):.1f} & "
                    f"{int(np.sum(v['n_adv_barred']))}/{int(np.sum(v['n_adv']))} & "
                    f"{mean(v['honest_false_excl']):.1f} & {asr} \\\\")
    write_table("tab_detail", rows, out)
    put("attackstart", cfg["attack_start"])
    for mk, key in (("fedavg_random", "rand"), ("atef_norep", "norep"), ("ra_gm", "gm"), ("ra_mkrum", "mkrum")):
        put(f"asr20-{key}", mean(B["backdoor"][mk]["asr"]), 0)
    put("asr-clean", mean(B["backdoor_clean"]["atef_dt"]["asr"]), 1)

    RA = [("atef_dt", r"\textbf{ATEF-DT}", "score", "reputation-weighted, clipped"),
          ("atef_norep", "ATEF-DT without reputation", "score", "clipped mean"),
          ("atef_norep_gm", "ATEF-DT without reputation", "score", "clipped geometric median"),
          ("ra_cwmed", "Coordinate-wise median~\\cite{yin2018byzantine}", "uniform", "median"),
          ("ra_trimmed", "Trimmed mean~\\cite{yin2018byzantine}", "uniform", "trimmed mean"),
          ("ra_mkrum", "Multi-Krum~\\cite{blanchard2017machine}", "uniform", "Multi-Krum"),
          ("ra_gm", "Geometric median~\\cite{pillutla2022robust}", "uniform", "geometric median")]
    rows = []
    for mk, name, selr, aggr in RA:
        cl = M["atef_dt"] if mk == "atef_dt" else B["methods"]["clean"][mk]
        sfv, lfv = B["methods"]["sign_flip"][mk], B["methods"]["label_flip"][mk]
        rows.append(f"{name} & {selr} & {aggr} & {pm(cl['final_acc'], 1, 100)} & {pm(sfv['final_acc'], 1, 100)} & "
                    f"{pm(lfv['final_acc'], 1, 100)} & {mean(cl['E_total']):.1f} & {mean(cl['jain_all']):.3f} \\\\")
        key = mk.replace("atef_", "").replace("ra_", "").replace("_", "")
        if mk != "atef_dt":
            put(f"cl-{key}", 100 * mean(cl["final_acc"]), 1)
            put(f"sf20-{key}", 100 * mean(sfv["final_acc"]), 1)
            put(f"lf20-{key}", 100 * mean(lfv["final_acc"]), 1)
    write_table("tab_robust", rows, out)

    # accuracy at 30 % and 40 % adversaries (outside the domain of Assumption 1)
    HK = [("atef_dt", r"\textbf{ATEF-DT}"), ("atef_norep", "ATEF-DT without reputation (clipped mean)"),
          ("ra_gm", "Geometric median"), ("ra_mkrum", "Multi-Krum"), ("fedavg_random", "FedAvg-Random")]
    cols = [c for c in ("sign_flip|0.3", "sign_flip|0.4", "label_flip|0.3", "label_flip|0.4") if c in B["high"]]
    rows = []
    for mk, name in HK:
        rows.append(f"{name} & " + " & ".join(pm(B["high"][c][mk]["final_acc"], 1, 100) for c in cols) + r" \\")
        for c in cols:
            a_, fr_ = c.split("|")
            key = mk.replace("atef_", "").replace("ra_", "").replace("fedavg_", "").replace("_", "")
            put(f"hi-{ASHORT[a_]}{int(round(float(fr_) * 100))}-{key}", 100 * mean(B["high"][c][mk]["final_acc"]), 1)
    write_table("tab_high", rows, out)

    # ablation C1-C4
    AB = [("ea_fl", "C1: energy-greedy selection (EA-FL)"),
          ("energy_fair", "C2: C1 + fairness factor and twin screen"),
          ("ef_trust", "C3: C2 + reputation layer"),
          ("atef_dt", r"\textbf{C4: C3 + adaptive quantization (ATEF-DT)}")]
    rows = []
    for mk, name in AB:
        v, lfv = M[mk], B["methods"]["label_flip"][mk]
        rows.append(f"{name} & {pm(v['final_acc'], 2, 100)} & {pm(v['E_total'], 2)} & {pm(v['jain_all'], 3)} & "
                    f"{mb(np.add(v['bits_payload'], v['bits_meta'])):.1f} & {pm(lfv['final_acc'], 1, 100)} \\\\")
    write_table("tab_ablation", rows, out)
    put("lf20-eftrust", 100 * mean(B["methods"]["label_flip"]["ef_trust"]["final_acc"]), 1)

    # ------------------------------------------------------------ trust sensitivity
    rows = []
    for key, v in T["grid"].items():
        mu, rm = key.split("|")
        ks = exclusion_bound(1.0, float(rm), float(mu))
        lfv = v["label_flip"]
        mark = r"$^{\ast}$" if (abs(float(mu) - cfg["mu"]) < 1e-9 and abs(float(rm) - cfg["rho_min"]) < 1e-9) else ""
        rows.append(f"{float(mu):.1f}{mark} & {float(rm):.2f} & {ks} & {mean(v['clean']['honest_false_excl']):.1f} & "
                    f"{pm(lfv['final_acc'], 1, 100)} & {mean(lfv['mal_weight_share']):.1f} & "
                    f"{int(np.sum(lfv['n_adv_barred']))}/{int(np.sum(lfv['n_adv']))} & "
                    f"{mean(lfv['honest_false_excl']):.1f} \\\\")
    write_table("tab_trustsens", rows, out)
    for onset, dct in T["onset"].items():
        for a_, v in dct.items():
            put(f"on{onset}-{ASHORT[a_]}-acc", 100 * mean(v["final_acc"]), 1)
            put(f"on{onset}-{ASHORT[a_]}-malw", mean(v["mal_weight_share"]), 1)
            put(f"on{onset}-{ASHORT[a_]}-bar", int(np.sum(v["n_adv_barred"])))
    put("wref-fe", mean(T["weighted_ref"]["clean"]["honest_false_excl"]), 0)
    put("wref-acc", 100 * mean(T["weighted_ref"]["clean"]["final_acc"]), 1)
    put("wref-jain", mean(T["weighted_ref"]["clean"]["jain_all"]), 3)

    # ------------------------------------------------------------ twin functional test
    rows = []
    for key, name in (("twin", r"\textbf{Twin forecast (ATEF-DT)}"), ("instant", "Instantaneous battery"),
                      ("none", "No reserve screen")):
        v = TW[key]
        rows.append(f"{name} & {pm(v['fl_breach'], 1)} & {pm(v['depleted'], 1)} & {mean(v['n_tx']):.0f} & "
                    f"{pm(v['jain_all'], 3)} & {pm(v['final_acc'], 1, 100)} \\\\")
        put(f"tw-breach-{key}", mean(v["fl_breach"]), 1)
        put(f"tw-dep-{key}", mean(v["depleted"]), 1)
        put(f"tw-jain-{key}", mean(v["jain_all"]), 3)
        put(f"tw-acc-{key}", 100 * mean(v["final_acc"]), 1)
    write_table("tab_twin", rows, out)
    put("tw-mae-level", mean(TW["twin"]["twin_mae"]), 3)
    put("tw-mae-ar", mean(TW["ar"]["twin_mae"]), 3)
    put("tw-mae-hold", mean(TW["twin"]["hold_mae"]), 3)
    put("tw-breach-ar", mean(TW["ar"]["fl_breach"]), 1)
    for sdv, dct in TW["noise"].items():
        k = sdv.replace(".", "")
        put(f"tw-mae-level-{k}", mean(dct["level"]["twin_mae"]), 3)
        put(f"tw-mae-ar-{k}", mean(dct["ar"]["twin_mae"]), 3)
        put(f"tw-mae-hold-{k}", mean(dct["level"]["hold_mae"]), 3)
    for k_, v_ in TW["config"].items():
        put("twc-" + k_.replace("_", ""), v_)

    # ------------------------------------------------------------ scalability and overhead
    rows = []
    for P, dct in SC.items():
        t = dct["atef_dt"]["timing"]
        tot = t["score"] + t["gm"] + t["rep"] + t["agg"]
        kk = int(round(int(P) * cfg["S_max"] / cfg["P"]))
        rows.append(f"{P} & {kk} & {t['score']:.2f} & {t['gm']:.2f} & {t['iters']:.1f} & {t['rep']:.2f} & "
                    f"{t['agg']:.2f} & {tot:.2f} \\\\")
        put(f"t-total-{P}", tot, 1)
        put(f"t-gm-{P}", t["gm"], 1)
        for mk in dct:
            put(f"sc-jain-{SHORT[mk]}-{P}", mean(dct[mk]["jain_all"]), 3)
            put(f"sc-acc-{SHORT[mk]}-{P}", 100 * mean(dct[mk]["final_acc"]), 1)
            put(f"sc-E-{SHORT[mk]}-{P}", mean(dct[mk]["E_total"]), 0)
        put(f"sc-sav-full-{P}", 100 * (1 - mean(dct["atef_dt"]["E_total"]) / mean(dct["fedavg_full"]["E_total"])), 1)
        put(f"sc-sav-rand-{P}", 100 * (1 - mean(dct["atef_dt"]["E_total"]) / mean(dct["fedavg_random"]["E_total"])), 1)
    write_table("tab_overhead", rows, out)
    put("pmax", max(int(P) for P in SC))

    # ------------------------------------------------------------ sensitivity to alpha and eta_PA
    rows = []
    for al, dct in SE["alpha"].items():
        e = {mk: mean(dct[mk]["E_total"]) for mk in dct}
        share = 100 * mean(dct["fedavg_full"]["E_comp"]) / e["fedavg_full"]
        expo = int(round(np.log10(float(al))))
        rows.append(f"$10^{{{expo}}}$ & {share:.1f} & {e['fedavg_full']:.2f} & {e['fedavg_random']:.2f} & "
                    f"{e['ea_fl']:.2f} & {e['atef_dt']:.2f} & {100 * (1 - e['atef_dt'] / e['fedavg_full']):.1f} & "
                    f"{100 * (1 - e['atef_dt'] / e['fedavg_random']):.1f} & {mean(dct['atef_dt']['jain_all']):.3f} \\\\")
        k = f"a{-expo}"
        put(f"sav-full-{k}", 100 * (1 - e["atef_dt"] / e["fedavg_full"]), 1)
        put(f"sav-rand-{k}", 100 * (1 - e["atef_dt"] / e["fedavg_random"]), 1)
        put(f"compshare-{k}", share, 1)
        put(f"E-atef-{k}", e["atef_dt"], 2)
        put(f"E-eafl-{k}", e["ea_fl"], 2)
        put(f"jain-atef-{k}", mean(dct["atef_dt"]["jain_all"]), 3)
        put(f"jain-eafl-{k}", mean(dct["ea_fl"]["jain_all"]), 3)
    write_table("tab_alpha", rows, out)
    for ep, dct in SE["eta_pa"].items():
        k = ep.replace(".", "")
        e = {mk: mean(dct[mk]["E_total"]) for mk in dct}
        put(f"sav-full-eta{k}", 100 * (1 - e["atef_dt"] / e["fedavg_full"]), 1)
        put(f"sav-rand-eta{k}", 100 * (1 - e["atef_dt"] / e["fedavg_random"]), 1)

    # ------------------------------------------------------------ write values.tex
    lines = ["% Generated by experiments/make_values.py -- do not edit by hand.",
             f"% {len(VAL)} values from {res.name}/ ({len(meta['seeds'])} seeds)."]
    lines += [f"\\defval{{{k}}}{{{v}}}" for k, v in sorted(VAL.items())]
    (out / "values.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'values.tex'} ({len(VAL)} values) and {len(list((out / 'tables').glob('*.tex')))} tables")


def VAL_float(key):
    return float(VAL[key])


if __name__ == "__main__":
    main()
