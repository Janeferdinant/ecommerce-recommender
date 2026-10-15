"""Simulated A/B test on the held-out period: how many customers would a real experiment need?

    python scripts/simulate_ab.py --dataset retail

THIS IS A SIMULATION UNDER STATED ASSUMPTIONS, NOT A DEPLOYED RESULT AND NOT A MEASURED BUSINESS EFFECT.

Control is the popularity baseline, treatment is the model with the best validation NDCG. Each held-out customer's
outcome under each arm comes from an offline replay: the customer "converts" if at least one product in that arm's top-10
list is a product they really bought in the test period (and had not bought before). A randomised experiment is then
simulated by drawing customers independently into two arms. Assumptions: a later purchase of a listed product is caused
by the list, customers do not influence each other, no novelty effect or drift, the population is the customers who
bought something new in the test window (so replay conversion rates are far higher than a live click or purchase rate
would be). The log tabulates required sample sizes, simulated power and the false-positive rate of an A/A test.
Writes docs/runs/<date>-ab-simulation-<dataset>.md/.json.
"""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from ecomrec.abtest import (  # noqa: E402
    aa_false_positive_rate,
    minimum_detectable_relative_uplift,
    required_n_per_arm_means,
    required_n_per_arm_proportions,
    simulate_power,
)
from ecomrec.datasets import load_bundle  # noqa: E402
from ecomrec.evaluation import evaluate_extended  # noqa: E402
from ecomrec.registry import LABELS, make_model, read_tuned  # noqa: E402

K = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["retail", "books"], default="retail")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "runs"))
    args = parser.parse_args()

    bundle = load_bundle(args.dataset, ROOT)
    split = bundle.split()
    tuned = read_tuned(ROOT / "configs" / f"tuned-{args.dataset}.json")
    candidates = {m: v["val_ndcg@10"] for m, v in tuned["models"].items() if "val_ndcg@10" in v and m != "popularity"}
    best = max(candidates, key=lambda m: candidates[m])
    params = dict(tuned["models"][best]["params"])
    models = [make_model("popularity", {}, args.seed), make_model(best, params, args.seed)]
    ev = evaluate_extended(split, models, k=K, seed=args.seed, beyond=False)
    y0 = ev.per_user["popularity"][f"hit_rate@{K}"]
    y1 = ev.per_user[best][f"hit_rate@{K}"]
    p0, p1 = float(y0.mean()), float(y1.mean())
    replay_uplift = p1 / p0 - 1.0
    has_rev = bundle.has_price
    r0 = ev.per_user["popularity"][f"revenue@{K}"] if has_rev else None
    r1 = ev.per_user[best][f"revenue@{K}"] if has_rev else None

    rng = np.random.default_rng(args.seed)
    scenarios: list[dict[str, Any]] = []
    for share in (1.0, 0.5, 0.25):  # fraction of the replay effect assumed to be real
        y1s = np.where(rng.random(len(y0)) < share, y1, y0)
        up = float(y1s.mean() / p0 - 1.0)
        n_req = required_n_per_arm_proportions(p0, up) if up > 0 else None
        power = simulate_power(y0, y1s, n_req, args.n_sims, args.seed) if n_req else None
        scenarios.append({"share_of_replay_effect": share, "relative_uplift": up, "n_per_arm": n_req,
                          "simulated_power_at_n": power})

    grid_p0 = sorted({0.02, 0.05, 0.10, round(p0, 3)})
    grid_up = [0.02, 0.05, 0.10, 0.20]
    size_table = {str(p): {str(u): required_n_per_arm_proportions(p, u) for u in grid_up if p * (1 + u) < 1} for p in grid_p0}
    n_grid = [100, 200, 400, 800, 1600, 3200]
    curve = [{"n_per_arm": n, "power_replay_effect": simulate_power(y0, y1, n, args.n_sims, args.seed),
              "power_half_effect": simulate_power(y0, np.where(np.random.default_rng(1).random(len(y0)) < 0.5, y1, y0), n,
                                                  args.n_sims, args.seed)} for n in n_grid]
    mde = {str(n): minimum_detectable_relative_uplift(p0, n) for n in (500, 1000, 2000, 5000, 10000)}
    aa = {str(n): aa_false_positive_rate(y0, n, args.n_sims, args.seed) for n in (200, 1000)}
    revenue: dict[str, Any] | None = None
    if r0 is not None and r1 is not None:
        sigma = float(np.sqrt((r0.var(ddof=1) + r1.var(ddof=1)) / 2))
        delta = float(r1.mean() - r0.mean())
        n_req = required_n_per_arm_means(sigma, delta) if delta > 0 else None
        revenue = {"control_mean": float(r0.mean()), "treatment_mean": float(r1.mean()), "sigma": sigma,
                   "delta": delta, "n_per_arm": n_req,
                   "simulated_power_at_n": simulate_power(r0, r1, n_req, args.n_sims, args.seed, kind="mean") if n_req else None,
                   "power_curve": [{"n_per_arm": n, "power": simulate_power(r0, r1, n, args.n_sims, args.seed, kind="mean")}
                                   for n in n_grid]}

    payload = {"date": args.date, "dataset": args.dataset, "label": "SIMULATION, not a deployed result",
               "control": "popularity", "treatment": best, "treatment_params": params, "k": K, "n_customers": len(y0),
               "replay_conversion": {"control": p0, "treatment": p1, "relative_uplift": replay_uplift},
               "scenarios": scenarios, "sample_size_table": size_table, "power_curve": curve,
               "minimum_detectable_uplift_at_p0": mde, "aa_false_positive_rate": aa, "revenue": revenue,
               "n_sims": args.n_sims, "seed": args.seed}

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    size_rows = "\n".join(
        f"| {float(p) * 100:.1f}% | " + " | ".join(f"{row[str(u)]:,}" if str(u) in row else "n/a" for u in grid_up) + " |"
        for p, row in size_table.items())
    scen_rows = "\n".join(
        f"| {pct(s['share_of_replay_effect'])} | {s['relative_uplift'] * 100:+.1f}% | "
        f"{('{:,}'.format(s['n_per_arm'])) if s['n_per_arm'] else 'n/a'} | "
        f"{pct(s['simulated_power_at_n']) if s['simulated_power_at_n'] is not None else 'n/a'} |" for s in scenarios)
    curve_rows = "\n".join(f"| {c['n_per_arm']:,} | {pct(c['power_replay_effect'])} | {pct(c['power_half_effect'])} |" for c in curve)
    mde_rows = "\n".join(f"| {int(n):,} | {v * 100:+.1f}% |" for n, v in mde.items())
    rev_text = ""
    if revenue:
        rev_text = f"""
## Revenue per customer (continuous outcome, same replay)

Mean replay Revenue@{K}: control {revenue['control_mean']:.3f}, treatment {revenue['treatment_mean']:.3f} GBP per customer
(sd about {revenue['sigma']:.2f}; the distribution is heavy-tailed, so a Welch test with a normal reference is only a
rough guide). Detecting the replay difference of {revenue['delta']:.3f} GBP needs about {revenue['n_per_arm']:,} customers per arm
(analytic); simulated power at that size: {pct(revenue['simulated_power_at_n'])}. A real test would be run on
actual order value, which is noisier than this proxy.

| Customers per arm | Simulated power (replay revenue effect) |
|---|---|
""" + "\n".join(f"| {c['n_per_arm']:,} | {pct(c['power']) } |" for c in revenue["power_curve"]) + "\n"
    log = f"""# A/B test simulation {args.date}: {bundle.label}

**This is a simulation on historical data under stated assumptions. It is not a deployed experiment and its outputs are not
measured business impact.**

- Command: `python scripts/simulate_ab.py --dataset {args.dataset}`; seed {args.seed}; {args.n_sims} simulated experiments per cell
- Control: popularity. Treatment: {LABELS[best]} (best validation NDCG@{K}); k = {K}
- Population: the {len(y0):,} held-out customers; each customer's outcome under each arm comes from an offline replay
- Outcome: the customer "converts" if at least one product in the arm's top {K} list is one they bought, new to them, in the
  test window. Replay conversion: control {pct(p0)}, treatment {pct(p1)} (relative {replay_uplift * 100:+.1f}%).
- Test: two-sided, alpha 0.05, pooled two-proportion z-test; power target 80%
- Assumptions: later purchases of listed products are attributed to the list; customers are independent; no novelty effect,
  seasonality or drift; the population only holds customers who bought something new, so these conversion rates are much
  higher than a live click-through or purchase rate. A live experiment needs its own baseline rate; the first table is
  therefore given for several baseline rates.

## Customers needed per arm (analytic) for a relative uplift in a conversion rate

| Baseline rate | +2% | +5% | +10% | +20% |
|---|---|---|---|---|
{size_rows}

## What if only part of the replay effect is real?

Each customer in the treatment arm keeps the replay outcome with the given probability and otherwise behaves as in the control arm.

| Share of the replay effect assumed real | Implied relative uplift | Customers per arm needed | Simulated power at that size |
|---|---|---|---|
{scen_rows}

## Simulated power against sample size (baseline {pct(p0)})

| Customers per arm | Power if the replay effect is real | Power if half of it is real |
|---|---|---|
{curve_rows}

## Smallest detectable relative uplift at baseline {pct(p0)}

| Customers per arm | Minimum detectable uplift (80% power) |
|---|---|
{mde_rows}

A/A check (both arms see popularity): false-positive rate {pct(aa['200'])} with 200 per arm and {pct(aa['1000'])} with 1,000 per arm; it should be close to 5%.
{rev_text}"""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.date}-ab-simulation-{args.dataset}"
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / f"{stem}.md").write_text(log, encoding="utf-8", newline="\n")
    print(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
