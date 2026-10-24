"""Precompute the data behind docs/index.html.

    python scripts/build_demo_data.py --dataset retail --seed 42
    python scripts/build_demo_data.py --dataset books --seed 42

Retail writes docs/demo-data.json (canonical) and docs/demo-data.js (the same payload wrapped in
``window.DEMO_DATA = ...`` so the page also works when opened from disk, where browsers block ``fetch``).
Goodbooks-10k writes docs/demo-data-books.js only. The files hold product or book titles, model-derived numbers
(neighbour scores and factors) and anonymous "Sample customer N" labels (retail); no ratings, user ids or customer ids
are written out. Uses the settings in configs/tuned-<dataset>.json.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecomrec.datasets import load_bundle  # noqa: E402
from ecomrec.demo_ref import recommend_from_exported, top_neighbours  # noqa: E402
from ecomrec.evaluate import item_prices  # noqa: E402
from ecomrec.registry import build_models, read_tuned  # noqa: E402

N_CUSTOMERS = 12
N_NEIGHBOURS = 30
TOP_K = 10
HISTORY_SHOWN = 12
CATALOGUE = {"retail": 800, "books": 600}
PRESET_TITLES = {
    "books": [
        ("Hunger Games trilogy", ["The Hunger Games (The Hunger Games, #1)", "Catching Fire (The Hunger Games, #2)",
                                  "Mockingjay (The Hunger Games, #3)"]),
        ("Lord of the Rings", ["The Fellowship of the Ring (The Lord of the Rings, #1)",
                               "The Two Towers (The Lord of the Rings, #2)",
                               "The Return of the King (The Lord of the Rings, #3)"]),
        ("Classic romance", ["Pride and Prejudice", "Emma", "Sense and Sensibility"]),
    ],
}
PRESET_POSITIONS = {"retail": [("Example basket 1", [0, 5, 12]), ("Example basket 2", [3, 40, 41, 90]),
                               ("Example basket 3", [7])]}


def sig(x: float, digits: int = 3) -> float:
    return round(float(x), digits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["retail", "books"], default="retail")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    bundle = load_bundle(args.dataset, ROOT)
    train, _ = bundle.split(args.test_fraction)
    tuned = read_tuned(ROOT / "configs" / f"tuned-{args.dataset}.json")
    models = {m.name: m.fit(train) for m in build_models(tuned, args.seed, bundle.has_price, bundle.excluded_models)}
    pop = models["popularity"]
    prices = item_prices(train) if bundle.has_price else {}

    item_ids = list(pop.ranking[: CATALOGUE[args.dataset]])
    pos = {it: i for i, it in enumerate(item_ids)}

    def idx(item: str) -> int:
        if item not in pos:
            pos[item] = len(item_ids)
            item_ids.append(item)
        return pos[item]

    customers: list[dict[str, Any]] = []
    if args.dataset == "retail":
        rng = np.random.default_rng(args.seed)
        sizes = train.groupby("user")["item"].nunique()
        candidates = sorted(sizes[(sizes >= 6) & (sizes <= 40)].index)
        chosen_users = [candidates[i] for i in sorted(rng.choice(len(candidates), N_CUSTOMERS, replace=False))]
        for n, user in enumerate(chosen_users, start=1):
            recent = (train[train["user"] == user].sort_values(["ts", "item"], ascending=[False, True])
                      .drop_duplicates("item")["item"].tolist())
            entry: dict[str, Any] = {"label": f"Sample customer {n}", "n_products": int(sizes[user]),
                                     "history": [idx(i) for i in recent[:HISTORY_SHOWN]], "recs": {}}
            for name, model in models.items():
                recs = []
                for item in model.recommend(user, TOP_K):
                    rec: dict[str, Any] = {"i": idx(item), "w": [[idx(s), sig(sh, 2)] for s, sh in
                                                                 model.explain_top(user, item, 3)]}
                    if name == "copurchase_price":
                        rec["f"] = sig(model.price_weight[model.item_index[item]], 4)
                    recs.append(rec)
                entry["recs"][name] = recs
            customers.append(entry)

    cat_cols = np.array([models["itemknn"].item_index[i] for i in pop.ranking[: CATALOGUE[args.dataset]]])
    block = np.ix_(cat_cols, cat_cols)
    n_knn = 40 if args.dataset == "retail" else N_NEIGHBOURS
    sources = {
        "itemknn": (lambda m: m.similarity, n_knn),
        "itemknn_recency": (lambda m: m.similarity, N_NEIGHBOURS),
        "ease": (lambda m: m.weights, N_NEIGHBOURS),
        "copurchase": (lambda m: m.assoc, N_NEIGHBOURS),
    }
    lists = {name: top_neighbours(get(models[name])[block], n) for name, (get, n) in sources.items() if name in models}
    als, bpr = models["als"], models["bpr"]
    data: dict[str, Any] = {
        "meta": {
            "dataset": bundle.label + (" (CC BY 4.0)" if args.dataset == "retail" else " (CC BY-SA 4.0)"),
            "dataset_key": args.dataset,
            "seed": args.seed,
            "models": list(models),
            "protocol": bundle.protocol_text,
            "note": "Independent study on public data. Illustrative recommendations only.",
        },
        "catalogue": list(range(len(pop.ranking[: CATALOGUE[args.dataset]]))),
        "lists": lists,
        "als": {"alpha": als.alpha, "reg": als.reg, "factors": np.round(als.item_factors[cat_cols], 3).tolist(),
                "gram": np.round(als.gram, 3).tolist()},
        "bpr": {"factors": np.round(bpr.item_factors[cat_cols], 3).tolist(),
                "bias": np.round(bpr.item_bias[cat_cols], 3).tolist()},
    }
    if bundle.has_price:
        priced = models["copurchase_price"]
        assert priced.beta == models["copurchase"].beta and priced.min_support == models["copurchase"].min_support
        data["price_weight"] = [sig(w, 4) for w in priced.price_weight[cat_cols]]
        data["meta"]["gamma"] = priced.gamma
    data["customers"] = customers

    names = bundle.names
    data["items"] = [[it, names.get(it, it), sig(prices[it], 2) if it in prices else None] for it in item_ids]

    if args.dataset == "retail":
        presets = [{"label": label, "chosen": chosen} for label, chosen in PRESET_POSITIONS["retail"]]
    else:
        title_pos = {data["items"][p][1]: p for p in range(len(data["catalogue"]))}
        presets = []
        for label, titles in PRESET_TITLES["books"]:
            missing = [t for t in titles if t not in title_pos]
            assert not missing, f"preset titles not in the demo catalogue: {missing}"
            presets.append({"label": label, "chosen": [title_pos[t] for t in titles]})
    data["presets"] = presets
    data["selfcheck"] = [{"chosen": p["chosen"], "expected": recommend_from_exported(data, p["chosen"], TOP_K)}
                         for p in presets]
    data["selfcheck"].append({"chosen": [7], "expected": recommend_from_exported(data, [7], TOP_K)})

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    text = json.dumps(data, separators=(",", ":"))
    if args.dataset == "retail":
        (docs / "demo-data.json").write_text(text + "\n", encoding="utf-8", newline="\n")
        (docs / "demo-data.js").write_text("window.DEMO_DATA=" + text + ";\n", encoding="utf-8", newline="\n")
    else:
        (docs / "demo-data-books.js").write_text("window.DEMO_DATA_BOOKS=" + text + ";\n", encoding="utf-8", newline="\n")
    print(f"dataset={args.dataset} items={len(item_ids)} customers={len(customers)} bytes={len(text):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
