"""Tune the study's models on a validation window inside the training period and write a dated log.

    python scripts/tune_models.py --dataset retail
    python scripts/tune_models.py --dataset books --models ease als          # a subset (merged into the tuned file)

The test period is never used. Retail: the training period (earliest 80% of purchase records) is split again in time
(earliest 80% fit, latest 20% validation). Goodbooks-10k subset: the same temporal rule on file order.
Selection rule: highest validation NDCG@10 (earliest grid point wins ties), except the price-aware co-purchase model,
whose price exponent is chosen for validation Revenue@10 with the other settings inherited from the tuned
co-purchase model. Writes configs/tuned-<dataset>.json and docs/runs/<date>-tuning-<dataset>[-<name>].md/.json.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecomrec.datasets import load_bundle  # noqa: E402
from ecomrec.evaluation import evaluate_extended  # noqa: E402
from ecomrec.neighbours import CoPurchaseModel  # noqa: E402
from ecomrec.registry import GRIDS, LABELS, MODEL_ORDER, SEEDED, make_model, read_tuned, write_tuned  # noqa: E402
from ecomrec.tuning import tune_model  # noqa: E402

K = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["retail", "books"], default="retail")
    parser.add_argument("--models", nargs="+", default=[m for m in MODEL_ORDER if m in GRIDS])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--name", default="", help="suffix for the log file name (for partial runs)")
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()

    started = datetime.now(UTC)
    bundle = load_bundle(args.dataset, ROOT)
    train, test = bundle.split()
    fit, val = bundle.validation_split(train)
    cfg_path = ROOT / "configs" / f"tuned-{args.dataset}.json"
    tuned = read_tuned(cfg_path) if cfg_path.exists() else {"dataset": args.dataset, "models": {}}
    tuned.update({"dataset": args.dataset, "seed": args.seed, "protocol": bundle.protocol_text})

    # reference rows: the untuned baselines on the same validation window
    reference = {}
    for name in ("popularity", "itemknn"):
        res = evaluate_extended((fit, val), [make_model(name, {}, args.seed)], k=K, beyond=False)
        reference[name] = next(iter(res.summary.values()))
        tuned["models"][name] = {"params": {}, "objective": f"ndcg@{K}", f"val_ndcg@{K}": reference[name][f"ndcg@{K}"]}

    tables, seconds = {}, {}
    for name in [m for m in MODEL_ORDER if m in args.models]:
        if name in bundle.excluded_models:
            print(f"skipping {name}: the dataset lacks the data it needs")
            continue
        spec = GRIDS[name]
        t0 = time.perf_counter()
        extra: dict = {}
        if name == "copurchase_price":
            base = tuned["models"].get("copurchase", {}).get("params", GRIDS["copurchase"]["defaults"])
            extra = {"beta": base["beta"], "min_support": base["min_support"]}

            def builder(extra: dict = extra, **p: float) -> CoPurchaseModel:
                return CoPurchaseModel(**extra, **p)
        else:
            def builder(name: str = name, **p: float):  # type: ignore[no-untyped-def]
                return make_model(name, p, args.seed if name in SEEDED else 0)

        best, table = tune_model(builder, spec["grid"], train, bundle.protocol,
                                 spec["objective"], k=K, seed=args.seed)
        seconds[name] = time.perf_counter() - t0
        row = next(r for r in table if all(r[n] == best[n] for n in best))
        params = {**extra, **best}
        if name == "als":
            params["alpha"] = GRIDS["als"]["defaults"]["alpha"]
        tuned["models"][name] = {
            "params": params, "objective": spec["objective"],
            f"val_{spec['objective']}": row[spec["objective"]], f"val_ndcg@{K}": row[f"ndcg@{K}"],
        }
        tables[name] = table
        print(f"{name}: {params} ({seconds[name]:.0f} s)", flush=True)

    tuned["updated"] = args.date
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    write_tuned(cfg_path, tuned)

    def fmt(v: object) -> str:
        return "none" if v is None else str(v)

    sections = []
    for name, table in tables.items():
        spec = GRIDS[name]
        keys = list(spec["grid"])
        header = "| " + " | ".join(keys) + f" | NDCG@{K} | Hit rate@{K} | " + ("Revenue@10 | " if "revenue@10" in table[0] else "") + " |"
        divider = "|" + "---|" * (header.count("|") - 1)
        rows = []
        for r in table:
            chosen = all(r[n] == tuned["models"][name]["params"][n] for n in keys)
            cells = [fmt(r[n]) for n in keys] + [f"{r[f'ndcg@{K}']:.4f}", f"{r[f'hit_rate@{K}']:.4f}"]
            if "revenue@10" in r:
                cells.append(f"{r['revenue@10']:.4f}")
            rows.append("| " + " | ".join(cells) + " | " + ("**selected**" if chosen else "") + " |")
        sections.append(f"## {LABELS[name]}\n\nObjective: validation {spec['objective']}. Time: {seconds[name]:.0f} s.\n\n"
                        f"{header}\n{divider}\n" + "\n".join(rows) + f"\n\nSelected: `{tuned['models'][name]['params']}`\n")
    log = f"""# Tuning run {args.date}: {bundle.label} ({", ".join(tables) or "baselines only"})

Hyperparameters were chosen on a validation window inside the training period. The test period was not used.

- Started: {started.strftime('%Y-%m-%d %H:%M UTC')}
- Command: `python scripts/tune_models.py --dataset {args.dataset} --models {' '.join(args.models)}`
- Protocol: {bundle.protocol_text}
- Fit window: {len(fit):,} records; validation window: {len(val):,} records ({val['user'].nunique():,} customers)
- Seed {args.seed} (only ALS and BPR use it); k = {K}
- Environment: Python {platform.python_version()}, {platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs, CPU only
- Selection rule: highest validation NDCG@{K}, earliest grid point wins ties (price-aware co-purchase: validation Revenue@{K})
- Validation NDCG@{K} of the untuned baselines: popularity {reference['popularity'][f'ndcg@{K}']:.4f}, item-based CF {reference['itemknn'][f'ndcg@{K}']:.4f}

{chr(10).join(sections)}
Grids are coarse and the differences between the better settings are small; read the selections as reasonable defaults, not sharp optima.
Settings at the edge of a grid are flagged in the README.
"""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.date}-tuning-{args.dataset}" + (f"-{args.name}" if args.name else "")
    (out / f"{stem}.md").write_text(log, encoding="utf-8", newline="\n")
    payload = {"date": args.date, "dataset": args.dataset, "seed": args.seed, "tuned": tuned["models"],
               "tables": tables, "seconds": {k: round(v, 1) for k, v in seconds.items()}}
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
