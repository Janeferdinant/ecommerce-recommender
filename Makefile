# Convenience targets. The same commands are listed in the README for platforms without make.
# Use the interpreter of your active virtual environment (override with: make PYTHON=python3.11 test).
PYTHON ?= python

.PHONY: install download download-books test lint typecheck tune tune-books evaluate evaluate-books study coldstart ab figures demo-data check all

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

download:
	$(PYTHON) scripts/download_data.py

download-books:
	$(PYTHON) scripts/download_data.py --dataset books

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck test

# Earlier (2026-10-14) three-model evaluation, kept reproducible.
tune-als-retail:
	$(PYTHON) scripts/tune_als.py --seed 42

evaluate:
	$(PYTHON) scripts/run_evaluation.py --seeds 42 43 44 45 46

# The study: eight models, two datasets (tuning writes configs/tuned-<dataset>.json).
tune:
	$(PYTHON) scripts/tune_models.py --dataset retail

tune-books:
	$(PYTHON) scripts/tune_models.py --dataset books

study:
	$(PYTHON) scripts/run_study.py --dataset retail --seeds 42 43 44 45 46
	$(PYTHON) scripts/run_study.py --dataset books --seeds 42 43 44 45 46

coldstart:
	$(PYTHON) scripts/run_coldstart.py --dataset retail
	$(PYTHON) scripts/run_coldstart.py --dataset books

ab:
	$(PYTHON) scripts/simulate_ab.py --dataset retail

figures:
	$(PYTHON) scripts/make_figures.py --date $(DATE)

demo-data:
	$(PYTHON) scripts/build_demo_data.py --dataset retail --seed 42
	$(PYTHON) scripts/build_demo_data.py --dataset books --seed 42

all: check download download-books study coldstart ab demo-data
