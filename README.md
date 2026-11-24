# ATEF-DT

Reference implementation and complete experiment pipeline for the article

> **ATEF-DT: Multi-Objective Energy, Fairness, and Trust Optimization for Federated Digital-Twin Healthcare**

ATEF-DT schedules battery-constrained clients of a federated-learning system with a single
weighted-product score over predicted round energy, participation history and reputation.
A patient-centric digital twin forecasts the monitoring energy a device must keep in
reserve; selected clients quantize their update with a per-client bit-width; the server
compares the received updates with a geometric-median reference and aggregates them with
reputation weights after norm clipping.

Every number, table and figure of the article is produced by the three scripts in
`experiments/` from the logs in `results/`. Nothing in the article is typed by hand.

## What is real and what is simulated

| Component | Status |
|---|---|
| Clinical data (Wisconsin Diagnostic Breast Cancer, 569 cases, 30 features) | real, loaded from scikit-learn |
| Model training (30-64-1 perceptron, NumPy) | real, executed in full |
| Client partition, devices, batteries, wireless channel | simulated from stated distributions |
| Energy | computed from the model of Section 3.3, **not measured on hardware** |
| Physiological streams for the twin test | synthetic |
| Adversaries | simulated (six attack models) |
| Server run time | measured wall-clock time of this code |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # 52 tests, a few seconds
python experiments/run_all.py --quick --out /tmp/atefdt_quick   # smoke test, about 20 s
python experiments/run_all.py             # full protocol of the article, about 3 min on one core
python experiments/make_values.py         # paper/values.tex and paper/tables/*.tex
python experiments/make_figures.py        # paper/figures/*.png (600 dpi) and figures/*.png
cd paper && sh build.sh main              # optional: rebuild the PDF (needs pdflatex and bibtex)
```

`make reproduce` runs the three experiment scripts in order; `make test`, `make quick`
and `make paper` are also available. A single experiment can be run with, for example,
`python experiments/run_all.py --only main byzantine`.

Running one method directly:

```python
from atefdt.config import load_config
from atefdt.simulator import run

cfg = load_config()                       # configs/default.yaml
r = run(cfg, "atef_dt", seed=0)
print(r["final_acc"], r["E_total"], r["jain_all"])
r = run(cfg, "atef_dt", seed=0, attack="label_flip", adv_frac=0.2)
print(r["final_acc"], r["mal_weight_share"])
```

## Main result (five seeds, no adversaries)

| Method | Accuracy (%) | Sensitivity (%) | Energy (J) | Jain index | Uplink (Mb) |
|---|---|---|---|---|---|
| FedAvg-Full | 97.02 ± 1.00 | 95.7 ± 1.1 | 52.14 ± 3.80 | 0.995 ± 0.007 | 63.6 |
| FedAvg-Random | 97.37 ± 1.07 | 95.7 ± 2.0 | 21.30 ± 1.56 | 0.963 ± 0.015 | 26.2 |
| EA-FL | 95.26 ± 1.82 | 93.3 ± 4.3 | 11.28 ± 1.53 | 0.441 ± 0.010 | 26.2 |
| FairFedCS | 97.54 ± 2.00 | 97.6 ± 2.4 | 21.63 ± 1.46 | 0.835 ± 0.052 | 26.2 |
| Energy+Trust | 95.79 ± 0.73 | 91.4 ± 5.5 | 11.66 ± 1.43 | 0.466 ± 0.032 | 26.2 |
| Energy+Fair | 96.84 ± 0.48 | 95.7 ± 1.1 | 19.27 ± 1.03 | 0.974 ± 0.004 | 26.2 |
| **ATEF-DT** | 96.67 ± 0.73 | 92.9 ± 2.9 | 17.72 ± 1.03 | 0.968 ± 0.004 | 3.6 |

Mean ± sample standard deviation over seeds 0–4, 20 clients, at most 8 per round, 50 rounds.
The full set of results, including the attack scenarios in which the method fails, is in the
article and in `results/`.

## Repository layout

```
atefdt/               the package
  config.py           all parameters (dataclass) and the YAML loader
  data.py             WDBC split, Dirichlet and bootstrap partitions
  model.py            perceptron with analytic gradient (NumPy)
  system.py           devices, Rayleigh block fading, energy model
  quantization.py     stochastic quantizer, stability surrogate, bit-width rule
  twin.py             Kalman twin, acuity, monitoring duty, synthetic patient
  selection.py        product score and all baseline selection rules
  trust.py            geometric median, cosine signal, reputation
  aggregation.py      reputation-weighted rule and robust-aggregation baselines
  attacks.py          six attack models
  simulator.py        run(): one method, one seed, all reported quantities
  metrics.py          Jain index, paired tests, JSON helpers
configs/default.yaml  the configuration of the article
experiments/
  run_all.py          all experiments -> results/*.json, results/logs/tx_*.csv
  make_values.py      results -> paper/values.tex, paper/tables/*.tex
  make_figures.py     results -> paper/figures/*.png (600 dpi), figures/*.png
tests/                unit and end-to-end tests
results/              stored results of the full protocol (regenerable)
figures/              PNG previews of the figures (paper/figures/fig_framework.drawio is the editable
                      source of Figure 1)
paper/                LaTeX source of the article (main.tex, sections, values.tex, tables, figures,
                      references.bib, REFERENCE_AUDIT.md, build.sh, flatten.py, texmf/ with the class files)
```

## From the article to the code

| Article | Experiment (`run_all.py --only ...`) | Output |
|---|---|---|
| Table 5, Fig. 2 (accuracy, convergence) | `main` | `results/main.json` |
| Table 6, Fig. 3 (energy) | `main` | `results/main.json`, `results/logs/tx_*.csv` |
| Table 7, Fig. 4 (bit-width policies, traffic) | `quant` | `results/quant.json` |
| Fig. 5, Table 8 (participation, device tiers) | `main` | `results/main.json` |
| Fig. 6 (fairness exponent, participation bias) | `fairness` | `results/fairness.json` |
| Fig. 7, Table 9 (multi-objective schedulers, rescaling) | `schedulers` | `results/schedulers.json` |
| Tables 10–15, Fig. 8 (attacks, robust aggregation) | `byzantine` | `results/byzantine.json` |
| Table 16 (reputation sensitivity, onset, weighted reference) | `trust` | `results/trust.json` |
| Table 17 (twin functional test) | `twin` | `results/twin.json` |
| Table 18, Fig. 9 (ablation) | `main`, `byzantine` | as above |
| Fig. 10, Table 19, Fig. 11a (scalability, server time) | `scalability` | `results/scalability.json` |
| Table 20, Fig. 11b (energy-parameter sensitivity) | `sensitivity` | `results/sensitivity.json` |
| learning-rate selection, outage rate, data statistics | `setup` | `results/setup.json` |

Method keys accepted by `run()` are listed in `atefdt.simulator.METHODS`
(`fedavg_full`, `fedavg_random`, `ea_fl`, `fairfedcs`, `energy_trust`, `energy_fair`,
`ef_trust`, `atef_dt`, the bit-width variants `atef_q16`, `atef_q8`, `atef_qglobal`, the
schedulers `wsum`, `dpp`, `seq_ef`, `seq_fe`, the matched variants `atef_norep`,
`atef_norep_gm`, and the robust aggregators `ra_cwmed`, `ra_trimmed`, `ra_mkrum`, `ra_gm`).

## Per-transmission logs

`results/logs/tx_<method>.csv` holds one row per uplink transmission with the columns
`seed, round, client, N_p, kappa_p, h, bits_per_coord, E_comp_J, E_uplink_J, E_downlink_J,
accepted, malicious, coefficient`. The energy and traffic totals of the article are sums
over these rows.

## Reproducibility

* A seed fixes, through independent random streams, the partition, the initial model, the
  device parameters, the complete fading sequence and the placement of the adversaries.
  These are identical for every method run with that seed.
* Runs are deterministic: the same seed gives the same numbers.
* The stored `results/` were produced with Python 3.13, NumPy 2.5, SciPy 1.18,
  scikit-learn 1.9 and Matplotlib 3.11. Other versions that satisfy `requirements.txt`
  should give the same results up to floating-point rounding; the test
  `test_fresh_run_agrees_with_the_shipped_results` checks this.
* The learning rate was selected on the validation split with FedAvg-Full
  (`results/setup.json`).

## Limitations

The energy figures are model outputs, not hardware measurements. The evaluation uses one
small tabular data set and one full-batch local step per round. The reputation layer fails
against colluding clients and when adversaries hold half of a round; the corresponding
results are part of the article and of `results/byzantine.json`. The twin layer is tested on
synthetic streams only. See Section 6.9 of the article.

## Data

The Wisconsin Diagnostic Breast Cancer data set (Wolberg, Mangasarian, Street and Street,
UCI Machine Learning Repository, doi:10.24432/C5DW2B) is distributed with scikit-learn and
is loaded with `sklearn.datasets.load_breast_cancer`; no download is needed.

## License and citation

The code is released under the MIT License (see `LICENSE`). Please cite the article; see
`CITATION.cff`.
#   A T E F - D T  
 