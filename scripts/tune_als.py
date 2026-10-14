"""Tune ALS (factors, regularisation, iterations) on a validation window.

    python scripts/tune_als.py --seed 42

The test period is never used: the training period (earliest 80% of purchase
records) is split again in time, the earliest 80% of that is used for fitting and
the latest 20% for validation. Writes docs/runs/<date>-als-tuning.md and .json.
Copy the winning values into ``src/ecomrec/config.py`` (ALS_PARAMS).
"""
import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecomrec.config import ALS_PARAMS  # noqa: E402
from ecomrec.data import load_clean  # noqa: E402
from ecomrec.split import temporal_split  # noqa: E402
from ecomrec.tuning import DEFAULT_GRID, tune_als, validation_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()

    started = datetime.now(UTC)
    inter, _ = load_clean(ROOT / "data" / "raw" / "online_retail_II.xlsx", ROOT / "data" / "cache")
    train, test = temporal_split(inter, args.test_fraction)
    fit, val = validation_split(train)

    t0 = time.perf_counter()
    best, table = tune_als(train, DEFAULT_GRID, seed=args.seed, k=args.k)
    seconds = time.perf_counter() - t0
    k = args.k

    rows = "\n".join(
        f"| {r['factors']} | {r['reg']} | {r['iterations']} | {r[f'precision@{k}']:.4f} | "
        f"{r[f'recall@{k}']:.4f} | {r[f'ndcg@{k}']:.4f} | "
        + ("**selected**" if all(r[n] == best[n] for n in best) else "")
        + " |"
        for r in table
    )
    log = f"""# ALS tuning run {args.date}

Hyperparameters were chosen on a validation window inside the training period.
The test period was not used (and the first test timestamp, {test['ts'].min()}, is after the validation window ends).

- Started: {started.strftime('%Y-%m-%d %H:%M UTC')}
- Command: `python scripts/tune_als.py --seed {args.seed}`
- Seed: {args.seed}; alpha held at {ALS_PARAMS['alpha']}
- Fit window: {len(fit):,} records, {fit['ts'].min()} to {fit['ts'].max()}
- Validation window: {len(val):,} records, {val['ts'].min()} to {val['ts'].max()}
- Grid: factors {DEFAULT_GRID['factors']}, regularisation {DEFAULT_GRID['reg']}, iterations {DEFAULT_GRID['iterations']} ({len(table)} fits)
- Selection rule: highest validation NDCG@{k} (earliest grid point wins ties)
- Wall time: {seconds:.0f} s on CPU

| factors | reg | iterations | Precision@{k} | Recall@{k} | NDCG@{k} | |
|---|---|---|---|---|---|---|
{rows}

Selected: {best}

The grid is coarse and validation differences between the better settings are small; treat the choice as a
reasonable default, not a sharp optimum.
"""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.date}-als-tuning.md").write_text(log, encoding="utf-8", newline="\n")
    payload = {"date": args.date, "seed": args.seed, "best": best, "table": table, "seconds": round(seconds, 1)}
    (out / f"{args.date}-als-tuning.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
