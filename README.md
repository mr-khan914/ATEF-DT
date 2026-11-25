# ATEF-DT: Adaptive Trust-Aware and Energy-Fair Federated Learning for Digital-Twin Healthcare

<p align="center">
  <img src="figures/architecture.png" alt="ATEF-DT framework" width="850"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-JKSUCIS%20(under%20review)-blue?style=flat-square" alt="Paper"/>
  <img src="https://img.shields.io/badge/Python-3.9+-green?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/NumPy%20%7C%20SciPy%20%7C%20scikit--learn-CPU%20only-orange?style=flat-square" alt="Dependencies"/>
  <img src="https://img.shields.io/badge/Tests-52%20passing-brightgreen?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

---

## 📋 Overview

This repository contains the reference implementation of **ATEF-DT** and the scripts that reproduce every number, table and figure of the article *"ATEF-DT: Multi-Objective Energy, Fairness, and Trust Optimization for Federated Digital-Twin Healthcare"*.

ATEF-DT schedules battery-constrained medical devices in federated learning. Each training round passes through four modules:

- **Patient-centric digital twin**: tracks the patient's state and reserves the energy needed for monitoring
- **Energy-aware client selection**: ranks devices by energy, participation and reputation
- **Local training and adaptive quantization**: each selected device trains locally and sends a compressed update
- **Credibility-based resilient aggregation**: combines the updates with reputation weights and filters unreliable ones
---

## 📁 Repository Structure

```
ATEF-DT/
├── README.md
├── requirements.txt
├── Makefile                  # one-line commands for every step
├── configs/
│   └── default.yaml          # all simulation and model settings
├── atefdt/                   # the method and the simulator
│   ├── twin.py               # patient-centric digital twin
│   ├── system.py             # devices, channel and energy model
│   ├── selection.py          # client-selection rules
│   ├── quantization.py       # adaptive quantization
│   ├── trust.py              # credibility engine (reputation)
│   ├── aggregation.py        # aggregation rules
│   ├── attacks.py            # adversarial clients
│   ├── simulator.py          # one federated training run
│   └── ...                   # data, model, metrics, config
├── experiments/
│   ├── run_all.py            # runs all experiments
│   ├── make_values.py        # turns results into the article's numbers and tables
│   └── make_figures.py       # draws all figures
├── results/                  # stored results and per-transmission logs
├── figures/                  # preview figures
├── paper/                    # LaTeX sources of the article
└── tests/                    # 52 automated tests
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url> ATEF-DT
cd ATEF-DT
pip install -r requirements.txt
```

### 2. Check the installation

```bash
make test        # runs the 52 tests (a few seconds)
make quick       # short demo run with two seeds (about 20 seconds)
```

### 3. Reproduce the article

```bash
make reproduce   # runs everything, then writes the numbers, tables and figures
```

The full protocol takes about **three minutes on one CPU core**. Each seed fixes the data partition, the devices, the channel and the adversaries, so the results are identical on every run. Only the measured server run times depend on the machine.

### 4. Run the steps one by one

```bash
make results     # 1) run all experiments        -> results/
make values      # 2) numbers and table bodies   -> paper/values.tex, paper/tables/
make figures     # 3) figures                    -> paper/figures/ and figures/
make paper       # 4) compile the article (needs pdflatex and bibtex)
```

Without `make`, use the scripts directly, for example `python experiments/run_all.py`.

---

## ⚙️ Changing the Settings

All settings are in `configs/default.yaml`, for example the number of clients, rounds, participation budget, battery range and the selection weights. Edit the file and run `make reproduce` again. To keep the original results, write to a new folder:

```bash
python experiments/run_all.py --out my_results
python experiments/make_values.py --results my_results --out my_paper
python experiments/make_figures.py --results my_results --out my_figures
```


## 📜 Citation

```bibtex
@article{atefdt2026,
  title   = {ATEF-DT: Multi-Objective Energy, Fairness, and Trust Optimization
             for Federated Digital-Twin Healthcare},
  journal = {Journal of King Saud University - Computer and Information Sciences},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📄 License

This project is licensed under the MIT License; see [LICENSE](LICENSE) for details. The WDBC data set is from the UCI Machine Learning Repository and is distributed with scikit-learn.
