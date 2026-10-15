"""Evaluate all models on the untouched test window of one dataset and write a dated run log.

    python scripts/run_study.py --dataset retail --seeds 42 43 44 45 46
    python scripts/run_study.py --dataset books

Uses the settings in configs/tuned-<dataset>.json (chosen on a validation window inside the training period by
scripts/tune_models.py). Reports, for k = 10: ranking metrics, engagement and revenue proxies, beyond-accuracy
measures (coverage, novelty, intra-list diversity, popularity-bias curves), a paired bootstrap comparison of the two
models that were best on the validation window, and uplifts over popularity with 95% bootstrap intervals over
customers. Seeds only change ALS and BPR; the other models are deterministic and run once.

These are offline measurements on historical data, not measured business impact.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

from ecomrec.datasets import load_bundle  # noqa: E402
from ecomrec.evaluation import evaluate_extended  # noqa: E402
from ecomrec.registry import LABELS, SEEDED, build_models, read_tuned  # noqa: E402
from ecomrec.stats import bootstrap_uplift_ci, mean_sd, paired_bootstrap_difference  # noqa: E402


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
    parser.add_argument("--dataset", choices=["retail", "books"], default="retail")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--name", default="study")
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()
    k = args.k

    started = datetime.now(UTC)
    bundle = load_bundle(args.dataset, ROOT)
    train, test = bundle.split()
    tuned = read_tuned(ROOT / "configs" / f"tuned-{args.dataset}.json")

    t0 = time.perf_counter()
    summaries: dict[str, list[dict[str, float]]] = {}
    per_user: dict[str, list[dict[str, np.ndarray]]] = {}
    curves: dict[str, list[dict[str, list[float]]]] = {}
    evaluation = None
    for n, seed in enumerate(args.seeds):
        models = build_models(tuned, seed, bundle.has_price, bundle.excluded_models)
        if n > 0:
            models = [m for m in models if m.name in SEEDED]
        evaluation = evaluate_extended((train, test), models, k=k, seed=seed, genres=bundle.genres)
        for name, summary in evaluation.summary.items():
            summaries.setdefault(name, []).append(summary)
            per_user.setdefault(name, []).append(evaluation.per_user[name])
            curves.setdefault(name, []).append(evaluation.popularity_curves[name])
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    assert evaluation is not None
    seconds = time.perf_counter() - t0
    names = [m for m in LABELS if m in summaries]
    metrics = list(summaries[names[0]][0])

    agg = {m: {x: mean_sd([s[x] for s in summaries[m]]) for x in metrics} for m in names}
    mean_user = {m: {x: np.mean([pu[x] for pu in per_user[m]], axis=0) for x in per_user[m][0]} for m in names}
    mean_curves = {m: {key: np.nanmean([c[key] for c in curves[m]], axis=0).tolist() for key in ("rec_share", "recall")}
                   for m in names}

    boot_metrics = [x for x in (f"hit_rate@{k}", f"ndcg@{k}", f"precision@{k}", f"recall@{k}", f"revenue@{k}") if x in metrics]
    uplift: dict[str, dict[str, Any]] = {}
    for m in names:
        if m == "popularity":
            continue
        uplift[m] = {}
        for x in boot_metrics:
            base = agg["popularity"][x][0]
            lo, hi = bootstrap_uplift_ci(mean_user[m][x], mean_user["popularity"][x], n_boot=args.n_boot, seed=0)
            uplift[m][x] = {"uplift": agg[m][x][0] / base - 1.0, "ci95": [lo, hi]}

    val_ndcg = {m: tuned["models"][m].get(f"val_ndcg@{k}") for m in names if m in tuned["models"]}
    val_rank = sorted((m for m in val_ndcg if val_ndcg[m] is not None), key=lambda m: -val_ndcg[m])
    test_rank = sorted(names, key=lambda m: -agg[m][f"ndcg@{k}"][0])
    best_two = val_rank[:2]
    comparison: dict[str, Any] = {}
    for x in [f"ndcg@{k}", f"hit_rate@{k}", f"revenue@{k}", f"novelty@{k}", f"diversity@{k}"]:
        if x in metrics:
            comparison[x] = paired_bootstrap_difference(mean_user[best_two[0]][x], mean_user[best_two[1]][x],
                                                        n_boot=2000, seed=0)

    payload = {
        "date": args.date, "dataset": args.dataset, "started_utc": started.isoformat(timespec="seconds"),
        "seeds": args.seeds, "k": k, "n_boot": args.n_boot, "protocol": bundle.protocol_text,
        "tuned": {m: tuned["models"][m] for m in names if m in tuned["models"]},
        "data": {
            "interactions": int(len(bundle.inter)), "users": int(bundle.inter["user"].nunique()),
            "items": int(bundle.inter["item"].nunique()), "train_interactions": int(len(train)),
            "test_interactions": int(len(test)), "train_items": int(train["item"].nunique()),
            "first": str(bundle.inter["ts"].min()) if bundle.has_time else None,
            "last": str(bundle.inter["ts"].max()) if bundle.has_time else None,
            "cutoff": str(test["ts"].min()) if bundle.protocol == "temporal" and bundle.has_time else None,
        },
        "n_eval_users": len(evaluation.users), "meta": evaluation.meta,
        "mean_sd": {m: {x: {"mean": v[0], "sd": v[1]} for x, v in d.items()} for m, d in agg.items()},
        "uplift_vs_popularity": uplift, "popularity_curves": mean_curves,
        "best_two_by_validation": best_two, "validation_ranking": val_rank, "test_ranking": test_rank,
        "comparison_best_two": comparison, "seconds": round(seconds, 1), "code_version": git_head(),
        "per_seed": {m: summaries[m] for m in names},
    }

    def fmt(m: str, x: str, digits: int = 4) -> str:
        mean, sd = agg[m][x]
        return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd > 0 else "")

    def table(cols: list[tuple[str, str]], digits: int = 4) -> str:
        head = "| Model | " + " | ".join(c[0] for c in cols) + " |\n|---|" + "---|" * len(cols)
        rows = [f"| {LABELS[m]} | " + " | ".join(fmt(m, c[1], digits) for c in cols) + " |" for m in names]
        return head + "\n" + "\n".join(rows)

    accuracy = table([(f"Hit rate@{k}", f"hit_rate@{k}"), (f"NDCG@{k}", f"ndcg@{k}"), (f"Precision@{k}", f"precision@{k}"),
                      (f"Recall@{k}", f"recall@{k}")])
    beyond_cols = [(f"Coverage@{k}", f"coverage@{k}"), (f"Novelty@{k} (bits)", f"novelty@{k}"),
                   (f"Diversity@{k}", f"diversity@{k}")]
    if bundle.genres:
        beyond_cols.append((f"Genre diversity@{k}", f"genre_diversity@{k}"))
    beyond = table(beyond_cols)
    revenue = ""
    if bundle.has_price:
        revenue = ("\n## Revenue proxy (retail only)\n\n"
                   + table([(f"Revenue@{k} (GBP/customer)", f"revenue@{k}"),
                            ("Share of the oracle's revenue", f"revenue_capture@{k}"),
                            ("Revenue with test-period prices (check)", f"revenue_testprice@{k}")])
                   + f"\n\nMedian ratio of test-period to training-period unit price over the relevant items: "
                     f"{evaluation.meta['price_drift_median']:.3f}. The oracle picks the {k} most valuable of each customer's "
                     "relevant products. Revenue@k sums training-period median unit prices of the recommended products the "
                     "customer really bought in the test period; it ignores quantity and margin and cannot capture sales a "
                     "recommendation causes.\n")
    up_rows = []
    for m, d in uplift.items():
        cells = [f"{d[x]['uplift'] * 100:+.1f}% ({d[x]['ci95'][0] * 100:+.1f}% to {d[x]['ci95'][1] * 100:+.1f}%)"
                 for x in boot_metrics]
        up_rows.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    uplift_table = ("| Model | " + " | ".join(boot_metrics) + " |\n|---|" + "---|" * len(boot_metrics) + "\n"
                    + "\n".join(up_rows))
    cmp_rows = []
    for x, c in comparison.items():
        p = c["p_two_sided"]
        cmp_rows.append(f"| {x} | {c['mean_a']:.4f} | {c['mean_b']:.4f} | {c['diff']:+.4f} "
                        f"({c['diff_ci'][0]:+.4f} to {c['diff_ci'][1]:+.4f}) | "
                        f"{c['rel_diff'] * 100:+.1f}% ({c['rel_diff_ci'][0] * 100:+.1f}% to {c['rel_diff_ci'][1] * 100:+.1f}%) | "
                        f"{'< ' + format(1 / 2000, '.4f') if p <= 1 / 2000 else format(p, '.4f')} |")
    n_bins = len(mean_curves["popularity"]["rec_share"])
    share_rows = "\n".join(f"| {LABELS[m]} | " + " | ".join(f"{v * 100:.1f}" for v in mean_curves[m]["rec_share"]) + " |"
                           for m in names)
    recall_rows = "\n".join(f"| {LABELS[m]} | " + " | ".join("n/a" if np.isnan(v) else f"{v * 100:.1f}"
                                                            for v in mean_curves[m]["recall"]) + " |" for m in names)
    bin_head = "| Model | " + " | ".join(str(i + 1) for i in range(n_bins)) + " |\n|---|" + "---|" * n_bins
    tuned_rows = "\n".join(f"| {LABELS[m]} | `{tuned['models'][m]['params']}` | {tuned['models'][m].get(f'val_ndcg@{k}', float('nan')):.4f} |"
                           for m in names if m in tuned["models"])
    d = payload["data"]
    span = (f"{d['first'][:10]} to {d['last'][:10]}" if d["first"]
            else "not available (no timestamps; file order is used)")
    log = f"""# Study run {args.date}: {bundle.label}

Status: Verified (run {args.date}). Offline evaluation on historical data; not measured business impact.

- Started: {started.strftime('%Y-%m-%d %H:%M UTC')}
- Command: `python scripts/run_study.py --dataset {args.dataset} --seeds {' '.join(map(str, args.seeds))}`
- Code version: {payload['code_version']}
- Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, {platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs, CPU only
- Seeds: {args.seeds} (only ALS and BPR use the seed; the other models are deterministic and were run once)
- Wall time: {seconds:.0f} s for fitting and evaluating all models and seeds
- Protocol: {bundle.protocol_text}
- Settings: `configs/tuned-{args.dataset}.json`, selected on a validation window inside the training period

## Data and split

| Quantity | Value |
|---|---|
| Interactions | {d['interactions']:,} |
| Customers / items | {d['users']:,} / {d['items']:,} |
| Span | {span} |
| Train / test interactions | {d['train_interactions']:,} / {d['test_interactions']:,} |
| Cut-off (temporal protocol only) | {d['cutoff'] or 'row position (no timestamps)'} |
| Evaluation customers | {len(evaluation.users):,} |
| Training catalogue | {d['train_items']:,} items |
| Relevant test items outside the training catalogue (unreachable) | {evaluation.meta['unreachable_relevant_share'] * 100:.1f}% |

Evaluation customers were seen in training and have at least one test item that they had not interacted with in
training. Recommendations exclude everything the customer interacted with in training; the ground truth is those new
items.

## Tuned settings (validation window)

| Model | Settings | Validation NDCG@{k} |
|---|---|---|
{tuned_rows}

## Accuracy at k = {k} (test window; mean ± sample sd over seeds where the model is seeded)

{accuracy}

## Beyond accuracy at k = {k}

Coverage is the share of the training catalogue that appears in anyone's list. Novelty is the mean self-information
(-log2 of the share of customers who interacted with the item); higher means less popular items. Diversity is one minus
the mean pairwise cosine similarity (on customer-item interaction vectors from the training period) between the items in a
list; higher means more varied lists.{' Genre diversity is the mean pairwise Jaccard distance between genre sets.' if bundle.genres else ''}

{beyond}
{revenue}
## Uplift over popularity (95% paired bootstrap intervals over customers, {args.n_boot} resamples)

{uplift_table}

## Best two models, compared (paired bootstrap over customers, 2000 resamples)

The two models with the highest validation NDCG@{k} (chosen before looking at the test window) were
{LABELS[best_two[0]]} (A) and {LABELS[best_two[1]]} (B). Validation ranking: {', '.join(LABELS[m] for m in val_rank)}.
Test ranking by NDCG@{k}: {', '.join(LABELS[m] for m in test_rank[:3])}, ...

| Metric | A | B | A - B (95% CI) | Relative (95% CI) | p (two-sided bootstrap) |
|---|---|---|---|---|---|
{chr(10).join(cmp_rows)}

## Popularity bias

Items are split into {n_bins} equal-sized bins by training popularity (bin 1 = the best-selling tenth of the
catalogue). First table: share of all recommended slots per bin (%). Second table: share of the relevant (customer, item)
pairs of that bin that were found in the top {k} (%).

Share of recommended slots per popularity bin (%):

{bin_head}
{share_rows}

Recall of relevant items per popularity bin (%):

{bin_head}
{recall_rows}

Seed-to-seed variation is the sd shown above; the intervals reflect which customers are in the test window. One
split is used, so neither captures variation across time windows.
"""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.date}-{args.name}-{args.dataset}"
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")
    (out / f"{stem}.md").write_text(log, encoding="utf-8", newline="\n")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
