#!/usr/bin/env python3
"""Draw every result figure of the manuscript from results/*.json.

    python experiments/make_figures.py

600-dpi PNGs go to paper/figures/ (used by LaTeX) and 200-dpi previews to figures/.
Colours follow the method, never its rank, and every series also differs by
line style and marker, so the figures remain readable in greyscale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from atefdt.metrics import load  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # key: (label, colour, line style, marker)
    "atef_dt": ("ATEF-DT", "#2a78d6", "-", "o"),
    "fedavg_full": ("FedAvg-Full", "#eb6834", "--", "s"),
    "fedavg_random": ("FedAvg-Random", "#1baf7a", "-.", "^"),
    "ea_fl": ("EA-FL", "#eda100", ":", "v"),
    "fairfedcs": ("FairFedCS", "#e87ba4", (0, (5, 1)), "D"),
    "energy_trust": ("Energy+Trust", "#008300", (0, (3, 1, 1, 1)), "P"),
    "energy_fair": ("Energy+Fair", "#6250d6", (0, (1, 1)), "X"),
}
MAIN = ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "energy_trust", "energy_fair", "atef_dt"]
WIDE, COL = 7.2, 3.5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "axes.axisbelow": True,
})


def mean(x):
    return float(np.nanmean(np.asarray(x, float)))


def sd(x):
    return float(np.nanstd(np.asarray(x, float), ddof=1))


def line(ax, x, y, key, every=5, **kw):
    lab, col, ls, mk = STYLE[key]
    lw = 2.0 if key == "atef_dt" else 1.3
    ax.plot(x, y, color=col, ls=ls, marker=mk, markevery=every, lw=lw, label=lab,
            zorder=3 if key == "atef_dt" else 2, markeredgecolor="white", markeredgewidth=0.4, **kw)


def panel(ax, tag):
    ax.set_title(tag, loc="left", fontsize=8, fontweight="bold", pad=3)


def save(fig, name, out, prev):
    fig.savefig(out / f"{name}.png", dpi=600)     # used by LaTeX
    fig.savefig(prev / f"{name}.png", dpi=200)    # preview
    plt.close(fig)


def top_legend(fig, ax, ncol, y=1.02):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.6, columnspacing=1.2)


# ---------------------------------------------------------------- figures
def fig_convergence(M, out, prev):
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    r = np.arange(1, len(M["atef_dt"]["curve_loss"]) + 1)
    for mk in MAIN:
        line(axs[0], r, M[mk]["curve_loss"], mk)
        line(axs[1], r, 100 * np.asarray(M[mk]["curve_acc"]), mk)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Test loss (cross-entropy)")
    axs[1].set_ylabel("Test accuracy (%)")
    axs[1].set_ylim(86, 99)
    for ax, t in zip(axs, ("(a)", "(b)")):
        ax.set_xlabel("Communication round")
        ax.set_xlim(1, len(r))
        panel(ax, t)
    top_legend(fig, axs[0], 7, 0.98)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_convergence", out, prev)


def fig_energy(M, out, prev):
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    r = np.arange(1, len(M["atef_dt"]["curve_E_cum"]) + 1)
    for mk in MAIN:
        line(axs[0], r, M[mk]["curve_E_cum"], mk)
    axs[0].set_xlabel("Communication round")
    axs[0].set_ylabel("Cumulative device energy (J)")
    axs[0].set_xlim(1, len(r))
    axs[0].legend(loc="upper left", ncol=1, handlelength=2.6)
    panel(axs[0], "(a)")
    comp = [("E_comp", "Computation", "#3d4654", None), ("E_ul", "Uplink", "#8b95a5", "///"),
            ("E_dl", "Downlink reception", "#cfd5de", "...")]
    y = np.arange(len(MAIN))[::-1]
    left = np.zeros(len(MAIN))
    for key, lab, col, hatch in comp:
        vals = np.array([mean(M[mk][key]) for mk in MAIN])
        axs[1].barh(y, vals, left=left, color=col, edgecolor="white", linewidth=0.8, height=0.62, label=lab,
                    hatch=hatch)
        left += vals
    for yi, mk in zip(y, MAIN):
        axs[1].text(left[MAIN.index(mk)] + 0.6, yi, f"{mean(M[mk]['E_total']):.1f}", va="center", fontsize=7,
                    color=INK)
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([STYLE[mk][0] for mk in MAIN], color=INK)
    axs[1].set_xlabel("Device energy over 50 rounds (J)")
    axs[1].set_xlim(0, left.max() * 1.14)
    axs[1].grid(axis="y", visible=False)
    axs[1].legend(loc="lower right", handlelength=1.4)
    panel(axs[1], "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_energy", out, prev)


def fig_fairness(M, cfg, out, prev):
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    R = len(M["atef_dt"]["curve_jain"])
    r = np.arange(1, R + 1)
    q = cfg["S_max"] / cfg["P"]
    ax.plot(r, r * q / (r * q + 1 - q), color=MUTED, lw=0.9, ls=(0, (6, 2)), label="Uniform sampling (analytic)",
            zorder=1)
    for mk in MAIN:
        line(ax, r, M[mk]["curve_jain"], mk)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Jain participation index")
    ax.set_xlim(1, R)
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.6, columnspacing=1.0)
    save(fig, "fig_fairness", out, prev)


def fig_frontier(S, M, out, prev):
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    fam = [("product", "Weighted product (ATEF-DT)", "#2a78d6", "-", "o"),
           ("additive", "Weighted sum", "#e34948", "--", "s"),
           ("dpp", "Drift-plus-penalty", "#6250d6", "-.", "^"),
           ("seq_ef", "Sequential, energy then fairness", "#008300", ":", "P"),
           ("seq_fe", "Sequential, fairness then energy", "#eb6834", (0, (5, 1)), "X")]
    for key, lab, col, ls, mk in fam:
        pts = sorted(((mean(it["E_total"]), mean(it["jain_all"])) for it in S["frontier"][key]))
        e, j = zip(*pts)
        ax.plot(e, j, color=col, ls=ls, marker=mk, lw=2.0 if key == "product" else 1.2, label=lab,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3 if key == "product" else 2)
    singles = [("fairfedcs", S["frontier"]["fairfedcs"][0], (0.5, -0.045)), ("fedavg_random", M["fedavg_random"], (0.5, -0.05)),
               ("ea_fl", M["ea_fl"], (0.6, 0.03))]
    for mk, v, off in singles:
        lab, col, _, mark = STYLE[mk]
        e, j = mean(v["E_total"]), mean(v["jain_all"])
        ax.plot([e], [j], color=col, marker=mark, ms=6, ls="none", markeredgecolor=INK, markeredgewidth=0.5, zorder=4)
        ax.annotate(lab, (e, j), xytext=(e + off[0], j + off[1]), fontsize=7, color=INK, ha="left", va="center")
    ax.axhline(0.95, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(ax.get_xlim()[0] + 0.2, 0.957, "Jain = 0.95", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Device energy over 50 rounds (J)")
    ax.set_ylabel("Jain participation index")
    ax.set_xlim(9, 25.5)
    ax.set_ylim(0.38, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.6, columnspacing=1.0)
    save(fig, "fig_frontier", out, prev)


def fig_bias(F, out, prev):
    lam = sorted(F, key=float)
    x = [float(v) for v in lam]
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 2.25))
    axs[0].plot(x, [mean(F[k]["c_bound"]) for k in lam], color=MUTED, ls="--", marker="s",
                label=r"Bound $\zeta\,\|\hat{\pi}-n/N\|_1$", markeredgecolor="white", markeredgewidth=0.4)
    axs[0].plot(x, [mean(F[k]["c_part"]) for k in lam], color="#2a78d6", marker="o", lw=2.0,
                label=r"Measured $\|\hat{c}_{\mathrm{part}}\|$", markeredgecolor="white", markeredgewidth=0.4)
    axs[0].set_ylabel("Gradient-norm units")
    axs[0].set_yscale("log")
    axs[0].legend(loc="upper right")
    for ax, key, lab, sc in ((axs[1], "jain_all", "Jain participation index", 1.0),
                             (axs[2], "final_acc", "Test accuracy (%)", 100.0)):
        m = np.array([mean(F[k][key]) for k in lam]) * sc
        s = np.array([sd(F[k][key]) for k in lam]) * sc
        ax.errorbar(x, m, yerr=s, color="#2a78d6", marker="o", lw=2.0, capsize=2, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_ylabel(lab)
    for ax, t in zip(axs, ("(a)", "(b)", "(c)")):
        ax.set_xlabel(r"Fairness exponent $\lambda_F$")
        panel(ax, t)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_bias", out, prev)


def fig_bitwidth(Q, out, prev):
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.3), sharex=True)
    r = np.arange(1, len(Q["atef_dt"]["curve_psi"]) + 1)
    axs[0].plot(r, Q["atef_dt"]["curve_psi"], color="#2a78d6", lw=2.0)
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"Stability surrogate $\psi^{(r)}$")
    panel(axs[0], "(a)")
    for key, lab, col, ls, mk in (("atef_qglobal", "Global schedule", "#eb6834", "--", "s"),
                                  ("atef_dt", "Per-client rule (ATEF-DT)", "#2a78d6", "-", "o")):
        axs[1].plot(r, Q[key]["curve_mean_b"], color=col, ls=ls, marker=mk, markevery=5, label=lab,
                    lw=2.0 if key == "atef_dt" else 1.3, markeredgecolor="white", markeredgewidth=0.4)
    axs[1].set_ylabel("Mean bit-width (bits)")
    axs[1].set_xlabel("Communication round")
    axs[1].set_ylim(3, 17)
    axs[1].set_yticks([4, 8, 12, 16])
    axs[1].set_xlim(1, len(r))
    axs[1].legend(loc="upper right")
    panel(axs[1], "(b)")
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_bitwidth", out, prev)


def fig_reputation(B, cfg, out, prev):
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 2.35))
    R = cfg["R"]
    r = np.arange(1, R + 1)
    for ax, atk_name, t in ((axs[0], "label_flip", "(a) Label flipping"), (axs[1], "sign_flip", "(b) Sign flipping")):
        v = B["grid"][atk_name]["0.2"]
        ax.plot(r, v["curve_rho_honest"], color="#2a78d6", lw=2.0, label="Honest clients")
        ax.plot(r, v["curve_rho_byz"], color="#eb6834", lw=1.6, ls="--", label="Adversarial clients")
        ax.axhline(cfg["rho_min"], color=MUTED, lw=0.7, ls=(0, (2, 2)))
        ax.text(R - 1, cfg["rho_min"] + 0.025, r"$\rho_{\min}$", fontsize=7, color=MUTED, ha="right")
        ax.axvline(cfg["attack_start"] + 0.5, color=MUTED, lw=0.7, ls=":")
        ax.set_ylabel("Mean reputation")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right")
        panel(ax, t)
    curves = [("fedavg_random", B["methods"]["label_flip"]["fedavg_random"], None),
              ("norep", B["methods"]["label_flip"]["atef_norep"], ("ATEF-DT without reputation", "#6250d6", ":", "X")),
              ("atef_dt", B["methods"]["label_flip"]["atef_dt"], None)]
    for key, v, st in curves:
        y = 100 * np.asarray(v["curve_acc"])
        if st is None:
            line(axs[2], r, y, key)
        else:
            axs[2].plot(r, y, color=st[1], ls=st[2], marker=st[3], markevery=5, lw=1.3, label=st[0],
                        markeredgecolor="white", markeredgewidth=0.4)
    axs[2].axvline(cfg["attack_start"] + 0.5, color=MUTED, lw=0.7, ls=":")
    axs[2].set_ylabel("Test accuracy (%)")
    axs[2].set_ylim(50, 100)
    axs[2].legend(loc="lower right")
    panel(axs[2], "(c) Label flipping")
    for ax in axs:
        ax.set_xlabel("Communication round")
        ax.set_xlim(1, R)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_reputation", out, prev)


def fig_ablation(M, out, prev):
    keys = ["ea_fl", "energy_fair", "ef_trust", "atef_dt"]
    labs = ["C1", "C2", "C3", "C4"]
    cols = ["#9dbfe8", "#9dbfe8", "#9dbfe8", "#2a78d6"]
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 2.2))
    # (a) accuracy: points with error bars (a bar would need a zero baseline)
    m = np.array([100 * mean(M[q]["final_acc"]) for q in keys])
    s = np.array([100 * sd(M[q]["final_acc"]) for q in keys])
    for i in range(4):
        axs[0].errorbar([i], [m[i]], yerr=[s[i]], color=cols[i] if i == 3 else "#6f9bd6", marker="o", ms=6,
                        capsize=3, elinewidth=1.0, markeredgecolor="white", markeredgewidth=0.5)
        axs[0].text(i + 0.12, m[i], f"{m[i]:.1f}", ha="left", va="center", fontsize=7)
    axs[0].set_xticks(range(4))
    axs[0].set_xticklabels(labs)
    axs[0].set_xlim(-0.5, 3.7)
    axs[0].set_ylim(92, 99)
    axs[0].set_ylabel("Test accuracy (%)")
    panel(axs[0], "(a)")
    spec = (("E_total", "Device energy (J)", (0, 24), "{:.1f}"),
            ("jain_all", "Jain participation index", (0, 1.12), "{:.2f}"))
    for ax, (k, lab, lim, fmt), t in zip(axs[1:], spec, ("(b)", "(c)")):
        m = np.array([mean(M[q][k]) for q in keys])
        s = np.array([sd(M[q][k]) for q in keys])
        ax.bar(labs, m, yerr=s, color=cols, edgecolor="white", width=0.62,
               error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": MUTED})
        for i, (mi, si) in enumerate(zip(m, s)):
            ax.text(i, mi + si + 0.012 * (lim[1] - lim[0]), fmt.format(mi), ha="center", va="bottom", fontsize=7)
        ax.set_ylim(*lim)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        panel(ax, t)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_ablation", out, prev)


def fig_scalability(SC, out, prev):
    Ps = sorted(SC, key=int)
    x = [int(p) for p in Ps]
    keys = ["fedavg_full", "fedavg_random", "ea_fl", "fairfedcs", "atef_dt"]
    fig, axs = plt.subplots(1, 3, figsize=(WIDE, 2.35))
    for mk in keys:
        line(axs[0], x, [mean(SC[p][mk]["jain_all"]) for p in Ps], mk, every=1)
        line(axs[1], x, [mean(SC[p][mk]["E_total"]) / int(p) for p in Ps], mk, every=1)
        line(axs[2], x, [100 * mean(SC[p][mk]["final_acc"]) for p in Ps], mk, every=1)
    axs[0].set_ylabel("Jain participation index")
    axs[1].set_ylabel("Device energy per client (J)")
    axs[2].set_ylabel("Test accuracy (%)")
    axs[0].set_ylim(0.3, 1.03)
    axs[1].set_ylim(0, None)
    axs[2].set_ylim(92, 99)
    for ax, t in zip(axs, ("(a)", "(b)", "(c)")):
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.minorticks_off()
        ax.set_xlabel("Number of clients $P$")
        panel(ax, t)
    top_legend(fig, axs[0], 5, 0.98)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_scalability", out, prev)


def fig_overhead(SC, SE, out, prev):
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.4))
    Ps = sorted(SC, key=int)
    x = [int(p) for p in Ps]
    comp = (("gm", "Geometric median", "#2a78d6", "-", "o"), ("rep", "Reputation update", "#eb6834", "--", "s"),
            ("agg", "Clipping and aggregation", "#1baf7a", "-.", "^"), ("score", "Selection score", "#6250d6", ":", "D"))
    for key, lab, col, ls, mk in comp:
        axs[0].plot(x, [SC[p]["atef_dt"]["timing"][key] for p in Ps], color=col, ls=ls, marker=mk, label=lab,
                    markeredgecolor="white", markeredgewidth=0.4)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([str(v) for v in x])
    axs[0].minorticks_off()
    axs[0].set_xlabel("Number of clients $P$")
    axs[0].set_ylabel("Server time per round (ms)")
    axs[0].legend(loc="upper left")
    panel(axs[0], "(a)")
    al = sorted(SE["alpha"], key=float)
    xa = [float(a) for a in al]
    e = {mk: np.array([mean(SE["alpha"][a][mk]["E_total"]) for a in al]) for mk in SE["alpha"][al[0]]}
    axs[1].plot(xa, 100 * (1 - e["atef_dt"] / e["fedavg_full"]), color="#eb6834", ls="--", marker="s",
                label="Relative to FedAvg-Full", markeredgecolor="white", markeredgewidth=0.4)
    axs[1].plot(xa, 100 * (1 - e["atef_dt"] / e["fedavg_random"]), color="#1baf7a", ls="-.", marker="^",
                label="Relative to FedAvg-Random", markeredgecolor="white", markeredgewidth=0.4)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"Computation energy coefficient $\alpha$ (J per multiply-accumulate)")
    axs[1].set_ylabel("Energy saving of ATEF-DT (%)")
    axs[1].set_ylim(0, 100)
    axs[1].legend(loc="lower left")
    panel(axs[1], "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_overhead", out, prev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--preview", default=str(ROOT / "figures"))
    args = ap.parse_args()
    res, out, prev = Path(args.results), Path(args.out), Path(args.preview)
    out.mkdir(parents=True, exist_ok=True)
    prev.mkdir(parents=True, exist_ok=True)
    cfg = load(res / "meta.json")["config"]
    M, Q, F, S = (load(res / f"{n}.json") for n in ("main", "quant", "fairness", "schedulers"))
    B, SC, SE = (load(res / f"{n}.json") for n in ("byzantine", "scalability", "sensitivity"))
    fig_convergence(M, out, prev)
    fig_energy(M, out, prev)
    fig_fairness(M, cfg, out, prev)
    fig_frontier(S, M, out, prev)
    fig_bias(F, out, prev)
    fig_bitwidth(Q, out, prev)
    fig_reputation(B, cfg, out, prev)
    fig_ablation(M, out, prev)
    fig_scalability(SC, out, prev)
    fig_overhead(SC, SE, out, prev)
    print(f"wrote {len(list(out.glob('fig_*.png')))} figures to {out}")


if __name__ == "__main__":
    main()
