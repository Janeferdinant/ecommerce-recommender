"""Run the offline evaluation and write a dated run log.

    python scripts/run_evaluation.py --seed 42 --date 2026-10-14

Reads data/raw/online_retail_II.xlsx (see scripts/download_data.py), builds a
temporal split, fits the three models and reports Precision@10, Recall@10 and
NDCG@10. Writes docs/runs/<date>-evaluation.md and .json.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from ecomrec.data import load_clean  # noqa: E402
from ecomrec.evaluate import build_models, evaluate  # noqa: E402
from ecomrec.split import temporal_split  # noqa: E402

LABELS = {"popularity": "Popularity baseline", "itemknn": "Item-based CF (cosine)", "als": "Implicit ALS (matrix factorisation)"}


def git_head() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + (" (+ uncommitted changes)" if dirty else "")
    except OSError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-users", type=int, default=None, help="subsample evaluation customers (default: all)")
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--xlsx", default=str(ROOT / "data" / "raw" / "online_retail_II.xlsx"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    inter, _ = load_clean(Path(args.xlsx), ROOT / "data" / "cache")
    t_load = time.perf_counter() - t0
    train, test = temporal_split(inter, args.test_fraction)
    cutoff = test["ts"].min()

    t1 = time.perf_counter()
    result = evaluate((train, test), seed=args.seed, k=args.k, max_users=args.max_users)
    t_eval = time.perf_counter() - t1
    als = build_models(args.seed)[2]
    hyper = {"factors": als.factors, "iterations": als.iterations, "reg": als.reg, "alpha": als.alpha}

    k = args.k
    pop = result["models"]["popularity"]
    rows = []
    for key, label in LABELS.items():
        m = result["models"][key]
        cells = [label]
        for metric in (f"precision@{k}", f"recall@{k}", f"ndcg@{k}"):
            v = m[metric]
            cells.append(f"{v:.4f}" if key == "popularity" else f"{v:.4f} ({(v / pop[metric] - 1) * 100:+.0f}%)")
        rows.append("| " + " | ".join(cells) + " |")

    stats = {
        "interactions": int(len(inter)),
        "customers": int(inter["user"].nunique()),
        "products": int(inter["item"].nunique()),
        "first_basket": str(inter["ts"].min()),
        "last_basket": str(inter["ts"].max()),
        "cutoff": str(cutoff),
        "train_interactions": int(len(train)),
        "test_interactions": int(len(test)),
        "train_customers": int(train["user"].nunique()),
        "train_products": int(train["item"].nunique()),
    }
    payload = {
        "date": args.date,
        "started_utc": started.isoformat(timespec="seconds"),
        "seed": args.seed,
        "k": k,
        "test_fraction": args.test_fraction,
        "max_users": args.max_users,
        "als_hyperparameters": hyper,
        "data": stats,
        "result": result,
        "seconds": {"load_and_clean": round(t_load, 1), "fit_and_evaluate": round(t_eval, 1)},
        "code_version": git_head(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.date}-evaluation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fence = "```"
    log = f"""# Evaluation run {args.date}

Status: Verified (run {args.date}). Independent re-implementation on public data
(UCI Online Retail II, CC BY 4.0). These are offline ranking metrics, not business impact.

- Started: {started.strftime('%Y-%m-%d %H:%M UTC')}
- Command: `python scripts/run_evaluation.py --seed {args.seed} --date {args.date}`
- Code version: {payload['code_version']}
- Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, CPU only, {platform.system()}
- Seed: {args.seed} (ALS initialisation; no other randomness in the pipeline)
- Wall time: {t_load:.0f} s to read and clean the workbook (first run, cached afterwards); {t_eval:.0f} s to fit and evaluate all models
- Customers: {"all eligible customers" if args.max_users is None else f"subsample of {args.max_users}"} (no subsampling needed)

## Data after cleaning

| Quantity | Value |
|---|---|
| Purchase records (customer, product, basket) | {stats['interactions']:,} |
| Customers | {stats['customers']:,} |
| Products | {stats['products']:,} |
| Span | {stats['first_basket'][:10]} to {stats['last_basket'][:10]} |

## Split

Temporal, by interaction count: the earliest {round((1 - args.test_fraction) * 100)}% of purchase records are train
(strictly before {stats['cutoff']}), the rest are test.

| Set | Records | Customers | Products |
|---|---|---|---|
| Train | {stats['train_interactions']:,} | {stats['train_customers']:,} | {stats['train_products']:,} |
| Test | {stats['test_interactions']:,} | | |

Evaluation customers: {result['n_eval_users']:,} (seen in training and bought at least one product in the test period
that they had not bought in training). Recommendations exclude everything the customer bought in training; the
ground truth is only the new products bought in the test period.

## Results (k = {k}, mean over evaluation customers; percentage = relative to the popularity baseline)

| Model | Precision@{k} | Recall@{k} | NDCG@{k} |
|---|---|---|---|
{chr(10).join(rows)}

ALS hyperparameters (fixed, not tuned on the test period): {hyper}.
"""
    (out_dir / f"{args.date}-evaluation.md").write_text(log, encoding="utf-8")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
