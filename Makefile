# ATEF-DT -- convenience targets.  PYTHON may be overridden: make PYTHON=python3.11 reproduce
PYTHON ?= python3

.PHONY: install test quick reproduce results values figures paper clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q tests

quick:            ## two seeds and reduced grids, written to /tmp (about 20 s)
	$(PYTHON) experiments/run_all.py --quick --out /tmp/atefdt_quick

results:          ## the full protocol of the article (about 3 min on one core)
	$(PYTHON) experiments/run_all.py

values:
	$(PYTHON) experiments/make_values.py

figures:
	$(PYTHON) experiments/make_figures.py

reproduce: results values figures   ## everything the article reports

paper: values figures               ## needs pdflatex and bibtex (class files are in paper/texmf)
	cd paper && sh build.sh main

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out paper/*.spl
