# Convenience targets. The same commands are listed in the README for platforms without make.
# Use the interpreter of your active virtual environment (override with: make PYTHON=python3.11 test).
PYTHON ?= python

.PHONY: install download test lint tune evaluate demo-data all

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

download:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

tune:
	$(PYTHON) scripts/tune_als.py --seed 42

evaluate:
	$(PYTHON) scripts/run_evaluation.py --seeds 42 43 44 45 46

demo-data:
	$(PYTHON) scripts/build_demo_data.py --seed 42

all: test lint download evaluate demo-data
