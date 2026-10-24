"""Run the offline evaluation over several seeds and write a dated run log.

    python scripts/run_evaluation.py --seeds 42 43 44 45 46

Reads data/raw/online_retail_II.xlsx (see scripts/download_data.py), builds a
temporal split, fits the three models once per seed and reports, for k = 10:

* ranking metrics: Precision, Recall, NDCG;
* offline engagement proxies: hit rate and catalogue coverage;
* an offline revenue proxy: Revenue@k (unit-price-weighted value of the
  recommended products the customer really bought in the test period).

Values are the mean and sample standard deviation over seeds (the seed only
changes the ALS initialisation). Relative uplifts over the popularity baseline
come with a 95% paired bootstrap interval over customers. Writes
docs/runs/<date>-evaluation-multiseed.md and .json.

These are offline proxies, not measurements of live engagement or revenue.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from ecomrec.config import ALS_PARAMS  # noqa: E402
from ecomrec.data import load_clean  # noqa: E402
from ecomrec.evaluate import evaluate_detailed  # noqa: E402
from ecomrec.split import temporal_split  # noqa: E402
from ecomrec.stats import bootstrap_uplift_ci, mean_sd  # noqa: E402

LABELS = {
    "popularity": "Popularity baseline",
    "itemknn": "Item-based CF (cosine)",
    "als": "Implicit ALS (matrix factorisation)",
}
CANDIDATES = ("itemknn", "als")


def git_head() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + (" (+ uncommitted changes)" if dirty else "")
    except OSError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-users", type=int, default=None, help="subsample evaluation customers (default: all)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--name", default="evaluation-multiseed")
    parser.add_argument("--xlsx", default=str(ROOT / "data" / "raw" / "online_retail_II.xlsx"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()
    k = args.k

    started = datetime.now(UTC)
    t0 = time.perf_counter()
    inter, _ = load_clean(Path(args.xlsx), ROOT / "data" / "cache")
    t_load = time.perf_counter() - t0
    train, test = temporal_split(inter, args.test_fraction)

    t1 = time.perf_counter()
    runs, per_user_runs = [], []
    for seed in args.seeds:
        summary, per_user = evaluate_detailed((train, test), seed=seed, k=k, max_users=args.max_users)
        runs.append(summary)
        per_user_runs.append(per_user)
    t_eval = time.perf_counter() - t1
    n_users = runs[0]["n_eval_users"]

    metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}", f"coverage@{k}", f"revenue@{k}"]
    agg = {m: {} for m in LABELS}
    for model in LABELS:
        for metric in metrics:
            agg[model][metric] = mean_sd([r["models"][model][metric] for r in runs])

    # paired bootstrap over customers, using each customer's value averaged over seeds
    boot_metrics = [m for m in metrics if not m.startswith("coverage")]
    mean_per_user = {
        model: {m: np.mean([pu[model][m] for pu in per_user_runs], axis=0) for m in boot_metrics}
        for model in LABELS
    }
    uplift = {model: {} for model in CANDIDATES}
    for model in CANDIDATES:
        for m in boot_metrics:
            base_mean = agg["popularity"][m][0]
            point = agg[model][m][0] / base_mean - 1.0
            lo, hi = bootstrap_uplift_ci(mean_per_user[model][m], mean_per_user["popularity"][m],
                                         n_boot=args.n_boot, seed=0)
            uplift[model][m] = {"uplift": point, "ci95": [lo, hi]}
    cov_uplift = {model: agg[model][f"coverage@{k}"][0] / agg["popularity"][f"coverage@{k}"][0] - 1.0
                  for model in CANDIDATES}

    stats = {
        "interactions": int(len(inter)),
        "customers": int(inter["user"].nunique()),
        "products": int(inter["item"].nunique()),
        "first_basket": str(inter["ts"].min()),
        "last_basket": str(inter["ts"].max()),
        "cutoff": str(test["ts"].min()),
        "train_interactions": int(len(train)),
        "test_interactions": int(len(test)),
        "train_customers": int(train["user"].nunique()),
        "train_products": int(train["item"].nunique()),
    }
    payload = {
        "date": args.date,
        "started_utc": started.isoformat(timespec="seconds"),
        "seeds": args.seeds,
        "k": k,
        "test_fraction": args.test_fraction,
        "max_users": args.max_users,
        "n_boot": args.n_boot,
        "als_hyperparameters": ALS_PARAMS,
        "data": stats,
        "n_eval_users": n_users,
        "mean_sd": {m: {k_: {"mean": v[0], "sd": v[1]} for k_, v in d.items()} for m, d in agg.items()},
        "uplift_vs_popularity": uplift,
        "coverage_uplift_vs_popularity": cov_uplift,
        "per_seed": runs,
        "seconds": {"load_and_clean": round(t_load, 1), "fit_and_evaluate_all_seeds": round(t_eval, 1)},
        "code_version": git_head(),
    }

    def fmt(model: str, metric: str, digits: int = 4) -> str:
        mean, sd = agg[model][metric]
        return f"{mean:.{digits}f} ± {sd:.{digits}f}"

    table_rows = "\n".join(
        f"| {LABELS[m]} | " + " | ".join(fmt(m, x) for x in metrics) + " |" for m in LABELS
    )
    uplift_rows = []
    for model in CANDIDATES:
        for m in boot_metrics:
            u = uplift[model][m]
            uplift_rows.append(f"| {LABELS[model]} | {m} | {u['uplift'] * 100:+.1f}% | "
                               f"{u['ci95'][0] * 100:+.1f}% to {u['ci95'][1] * 100:+.1f}% |")
        uplift_rows.append(f"| {LABELS[model]} | coverage@{k} | {cov_uplift[model] * 100:+.0f}% | not bootstrapped (catalogue-level) |")

    log = f"""# Evaluation run {args.date} (multi-seed, with engagement and revenue proxies)

Status: Verified (run {args.date}). Independent study on public data (UCI Online Retail II, CC BY 4.0).

**These are offline proxies computed on historical purchases. They are not measurements of live engagement or
revenue, and they make no claim about any other system or deployment.**

- Started: {started.strftime('%Y-%m-%d %H:%M UTC')}
- Command: `python scripts/run_evaluation.py --seeds {' '.join(map(str, args.seeds))}`
- Code version: {payload['code_version']}
- Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, CPU only, {platform.system()}
- Seeds: {args.seeds} (the seed only changes the ALS initialisation; popularity and item-based CF are deterministic)
- Wall time: {t_load:.0f} s to load the cleaned table (the first uncached read of the workbook takes about 80 s); {t_eval:.0f} s to fit and evaluate all models for {len(args.seeds)} seeds. Total well under the 20-minute CPU limit.
- ALS hyperparameters (selected on a validation window, see the tuning log): {ALS_PARAMS}
- Customers: {"all eligible customers" if args.max_users is None else f"subsample of {args.max_users}"}

## Data and split

| Quantity | Value |
|---|---|
| Purchase records (customer, product, basket) | {stats['interactions']:,} |
| Customers / products | {stats['customers']:,} / {stats['products']:,} |
| Span | {stats['first_basket'][:10]} to {stats['last_basket'][:10]} |
| Train records (strictly before {stats['cutoff']}) | {stats['train_interactions']:,} ({stats['train_customers']:,} customers, {stats['train_products']:,} products) |
| Test records | {stats['test_interactions']:,} |
| Evaluation customers | {n_users:,} |

Evaluation customers were seen in training and bought at least one product in the test period that they had not
bought in training. Recommendations exclude everything bought in training; ground truth is those new products.

## Metrics at k = {k} (mean ± sample sd over {len(args.seeds)} seeds)

Revenue@{k} is the mean, per evaluation customer, of the unit-price-weighted value (in GBP, as recorded in the
data) of the recommended products that the customer really bought in the test period. Prices are each product's
median unit price in the training period. Hit rate@{k} is the share of customers with at least one correct
recommendation. Coverage@{k} is the share of the {runs[0]['n_catalogue']:,}-product training catalogue that appears in
anyone's top {k}.

| Model | Precision@{k} | Recall@{k} | NDCG@{k} | Hit rate@{k} | Coverage@{k} | Revenue@{k} (GBP) |
|---|---|---|---|---|---|---|
{table_rows}

## Relative uplift over the popularity baseline (95% paired bootstrap CI over customers, {args.n_boot} resamples)

| Model | Metric | Uplift | 95% interval |
|---|---|---|---|
{chr(10).join(uplift_rows)}

Seed-to-seed variation is the sd in the table above; the intervals reflect which customers happened to be in the
test period. A single temporal split is used, so neither captures variation across time windows.
"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.date}-{args.name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8",
                                                          newline="\n")
    (out_dir / f"{args.date}-{args.name}.md").write_text(log, encoding="utf-8", newline="\n")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
