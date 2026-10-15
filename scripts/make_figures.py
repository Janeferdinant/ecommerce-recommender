"""Draw the README figures from the committed run logs (JSON files in docs/runs/).

    python scripts/make_figures.py --date 2026-10-14

Reads <date>-study-<dataset>.json, <date>-coldstart-<dataset>.json and <date>-ab-simulation-retail.json and writes
PNG files to docs/img/. Nothing is recomputed: every plotted number comes from those logs.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecomrec.registry import LABELS, MODEL_ORDER  # noqa: E402

# Okabe-Ito colour-blind-safe palette; markers differ as well so the lines never rely on colour alone
COLOURS = {"popularity": "#000000", "itemknn": "#0072B2", "itemknn_recency": "#56B4E9", "copurchase": "#009E73",
           "copurchase_price": "#CC79A7", "als": "#D55E00", "ease": "#E69F00", "bpr": "#999999"}
MARKERS = {"popularity": "o", "itemknn": "s", "itemknn_recency": "D", "copurchase": "^", "copurchase_price": "v",
           "als": "P", "ease": "*", "bpr": "X"}
TITLES = {"retail": "Online Retail II", "books": "Goodbooks-10k (subset)"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def shared_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True)
    parser.add_argument("--runs", default=str(ROOT / "docs" / "runs"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "img"))
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    studies = {d: load(runs / f"{args.date}-study-{d}.json") for d in ("retail", "books")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (d, s) in zip(axes, studies.items(), strict=True):
        for m in MODEL_ORDER:
            if m in s["popularity_curves"]:
                share = [v * 100 for v in s["popularity_curves"][m]["rec_share"]]
                ax.plot(range(1, len(share) + 1), share, marker=MARKERS[m], color=COLOURS[m], label=LABELS[m], lw=1.6, ms=5)
        ax.set_title(TITLES[d])
        ax.set_xlabel("Popularity bin of the recommended item (1 = best sellers)")
        ax.set_xticks(range(1, 11))
        style(ax)
    axes[0].set_ylabel("Share of recommended slots (%)")
    fig.suptitle("Popularity bias: where each model's recommendations come from", fontsize=11)
    shared_legend(fig, axes[0])
    fig.savefig(out / "popularity-bias.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (d, s) in zip(axes, studies.items(), strict=True):
        for m in MODEL_ORDER:
            if m in s["mean_sd"]:
                ax.scatter(s["mean_sd"][m]["novelty@10"]["mean"], s["mean_sd"][m]["ndcg@10"]["mean"], marker=MARKERS[m],
                           color=COLOURS[m], s=70, label=LABELS[m])
        ax.set_title(TITLES[d])
        ax.set_xlabel("Novelty@10 (bits; higher = less popular items)")
        style(ax)
    axes[0].set_ylabel("NDCG@10 (test window)")
    fig.suptitle("Accuracy against novelty", fontsize=11)
    shared_legend(fig, axes[0])
    fig.savefig(out / "accuracy-vs-novelty.png", dpi=130)
    plt.close(fig)

    cold = {d: load(runs / f"{args.date}-coldstart-{d}.json") for d in ("retail", "books")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (d, c) in zip(axes, cold.items(), strict=True):
        keys = [str(n) for n in c["n_values"]] + ["full"]
        for m in MODEL_ORDER:
            if m in c["results"]:
                ax.plot(range(len(keys)), [c["results"][m][k]["hit_rate"] * 100 for k in keys], marker=MARKERS[m],
                        color=COLOURS[m], label=LABELS[m], lw=1.6, ms=5)
        ax.set_xticks(range(len(keys)), [f"{k} item" + ("s" if k not in ("1",) else "") if k != "full" else "full history"
                                         for k in keys])
        ax.set_title(f"{TITLES[d]} ({c['n_cold_users']:,} simulated new customers)")
        style(ax)
    axes[0].set_ylabel("Hit rate@10 (%)")
    fig.suptitle("Cold start: hit rate with 1 to 3 known items", fontsize=11)
    shared_legend(fig, axes[0])
    fig.savefig(out / "cold-start.png", dpi=130)
    plt.close(fig)

    ab = load(runs / f"{args.date}-ab-simulation-retail.json")
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [c["n_per_arm"] for c in ab["power_curve"]]
    ax.plot(ns, [c["power_replay_effect"] * 100 for c in ab["power_curve"]], marker="o", color="#D55E00",
            label="replay effect is real")
    ax.plot(ns, [c["power_half_effect"] * 100 for c in ab["power_curve"]], marker="s", color="#0072B2",
            label="half of the replay effect is real")
    ax.axhline(80, color="#555555", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Customers per arm")
    ax.set_ylabel("Simulated power (%)")
    ax.set_title("A/B test simulation (not a deployed result)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "ab-power.png", dpi=130)
    plt.close(fig)
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
