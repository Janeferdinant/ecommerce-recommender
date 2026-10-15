# Fresh-clone check 2026-10-15: shop and books study

The branch `audit/2026-10-h` (commit 6164638, plus the demo-data ordering fix ddb536f) was cloned from GitHub into a new folder and a new Python 3.11 virtual environment, and the README commands were run. Long runs were replaced by documented smoke settings (stated below).

| Step | Result |
|---|---|
| `pip install -r requirements.txt -r requirements-dev.txt` | OK (pinned versions) |
| `python -m pytest -q` | 146 passed |
| `python -m ruff check .` | All checks passed! |
| `python -m mypy` | Success: no issues found in 18 source files |
| `python scripts/download_data.py` | OK: data verified (UCI Online Retail II, CC BY 4.0) |
| `python scripts/download_data.py --dataset books` | OK: data verified (Goodbooks-10k, CC BY-SA 4.0) |
| Tuning, smoke setting: `scripts/tune_models.py --models ease` on both datasets (not the full grids) | selected EASE regularisation 300 (books) and 1000 (retail), identical to the committed tuning logs |
| Study, smoke setting: `scripts/run_study.py --seeds 42` on both datasets (instead of five seeds) | all seed-42 values of all models and metrics identical to the committed logs (80 values for retail, 42 for books, 0 differences) |
| `scripts/run_coldstart.py --dataset books` (full) | results identical to the committed log |
| `scripts/build_demo_data.py` for retail and books | regenerated files byte-identical to the committed ones (after the ordering fix ddb536f; the first attempt differed only in the key order of the neighbour lists) |
| Not re-run in the clean clone | full five-seed studies, full tuning grids, retail cold start and A/B simulation (the retail logs were produced by the same code and verified in a fresh clone of the earlier branch; here they are carried over unchanged) |

Only timestamps and timings differ. The clone and venv are left in place (`work3\recommender-h-fresh`, `envs\ecommerce-recommender-h-fresh`).
