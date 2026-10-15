"""New-customer (cold start) evaluation: customers with only 1 to 3 known interactions.

    python scripts/run_coldstart.py --dataset retail
    python scripts/run_coldstart.py --dataset books

A random 25% of the evaluation customers are treated as new: all of their training interactions except the first n
(n = 1, 2, 3) are removed, every model is refitted on the reduced data and recommends from those n items only. The test
items are unchanged. "Full" is the same customers with their whole training history. One seed (ALS and BPR
initialisation and the choice of customers). Writes docs/runs/<date>-coldstart-<dataset>.md/.json.
"""
import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecomrec.coldstart import cold_start_eval  # noqa: E402
from ecomrec.datasets import load_bundle  # noqa: E402
from ecomrec.registry import GRIDS, LABELS, MODEL_ORDER, make_model, read_tuned  # noqa: E402
from ecomrec.stats import bootstrap_mean_ci  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["retail", "books"], default="retail")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cold-fraction", type=float, default=0.25)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()
    k = args.k

    started = datetime.now(UTC)
    bundle = load_bundle(args.dataset, ROOT)
    split = bundle.split()
    tuned = read_tuned(ROOT / "configs" / f"tuned-{args.dataset}.json")
    names = [m for m in MODEL_ORDER if m not in bundle.excluded_models]

    def factory(name: str):  # type: ignore[no-untyped-def]
        params = dict(tuned["models"].get(name, {}).get("params", GRIDS.get(name, {}).get("defaults", {})))
        return lambda: make_model(name, params, args.seed)

    t0 = time.perf_counter()
    res = cold_start_eval(split, {m: factory(m) for m in names}, tuple(args.n), k=k, seed=args.seed,
                          cold_fraction=args.cold_fraction)
    seconds = time.perf_counter() - t0
    keys = [*args.n, "full"]

    out = {m: {str(key): {"hit_rate": res["summary"][m][key][f"hit_rate@{k}"],
                          "hit_rate_ci95": list(bootstrap_mean_ci(res["per_user"][m][key][f"hit_rate@{k}"], 1000, 0)),
                          "ndcg": res["summary"][m][key][f"ndcg@{k}"],
                          "ndcg_ci95": list(bootstrap_mean_ci(res["per_user"][m][key][f"ndcg@{k}"], 1000, 0))}
               for key in keys} for m in names}
    head = "| Model | " + " | ".join(f"{key} known item{'s' if key != 1 else ''}" if key != "full" else "full history"
                                   for key in keys) + " |\n|---|" + "---|" * len(keys)

    def block(metric: str, label: str) -> str:
        rows = []
        for m in names:
            cells = [f"{out[m][str(key)][metric] * 100 if metric == 'hit_rate' else out[m][str(key)][metric]:.{1 if metric == 'hit_rate' else 4}f}"
                     f" ({out[m][str(key)][metric + '_ci95'][0] * (100 if metric == 'hit_rate' else 1):.{1 if metric == 'hit_rate' else 4}f} to "
                     f"{out[m][str(key)][metric + '_ci95'][1] * (100 if metric == 'hit_rate' else 1):.{1 if metric == 'hit_rate' else 4}f})"
                     for key in keys]
            rows.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
        return f"### {label}\n\n{head}\n" + "\n".join(rows)

    log = f"""# Cold-start run {args.date}: {bundle.label}

Status: Verified (run {args.date}). Offline simulation of new customers; not measured business impact.

- Started: {started.strftime('%Y-%m-%d %H:%M UTC')}
- Command: `python scripts/run_coldstart.py --dataset {args.dataset}`
- Protocol: {bundle.protocol_text}. Cold customers: {res['n_cold_users']:,} (a random {args.cold_fraction:.0%} of the evaluation customers, seed {args.seed}).
- Method: for n known items the customers' other training interactions are removed and every model is refitted;
  recommendations come from the n known items only (the models' basket mode). The first n items are the earliest
  by time, ties by item id. Test items are unchanged. One seed; wall time {seconds:.0f} s on CPU.
- Values: mean over the cold customers with 95% bootstrap intervals over customers (1,000 resamples), k = {k}.

{block('hit_rate', f'Hit rate@{k} (%)')}

{block('ndcg', f'NDCG@{k}')}

Read each column against the popularity row, which ignores the known items, and against the full-history column. Popularity
never uses the known items, so its small differences between columns come only from which items are excluded as already known.
"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.date}-coldstart-{args.dataset}"
    payload = {"date": args.date, "dataset": args.dataset, "seed": args.seed, "k": k, "n_cold_users": res["n_cold_users"],
               "n_values": args.n, "results": out, "seconds": round(seconds, 1)}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{stem}.md").write_text(log, encoding="utf-8", newline="\n")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
