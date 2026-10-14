"""Precompute the data behind docs/index.html.

    python scripts/build_demo_data.py --seed 42

Writes docs/demo-data.json (canonical) and docs/demo-data.js (the same payload
wrapped in ``window.DEMO_DATA = ...`` so the page also works when opened from
disk, where browsers block ``fetch``). Contains only product codes and public
product descriptions plus anonymous "Sample customer N" labels; customer ids
from the dataset are not written out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecomrec.data import load_clean  # noqa: E402
from ecomrec.evaluate import build_models  # noqa: E402
from ecomrec.split import temporal_split  # noqa: E402

N_CUSTOMERS = 12
N_CATALOGUE = 800
N_NEIGHBOURS = 40
TOP_K = 10
HISTORY_SHOWN = 12


def demo_recommend(data, chosen, k=TOP_K):
    """Reference implementation of the page's picker mode, run on the exported numbers.

    ``chosen`` holds catalogue positions. The page implements the same maths in
    JavaScript; the tests in the README compare the two.
    """
    cat = data["catalogue"]
    n = len(cat)
    chosen = sorted(set(chosen))
    out = {}

    # popularity: catalogue is stored in popularity order
    out["popularity"] = [{"i": cat[p], "w": None} for p in range(n) if p not in chosen][:k]

    # item-based CF from the stored nearest-neighbour lists
    scores, why = np.zeros(n), {}
    for c in chosen:
        for j, s in data["knn"][c]:
            if j in chosen:
                continue
            scores[j] += s
            if j not in why or s > why[j][1]:
                why[j] = (c, s)
    order = [j for j in np.lexsort((np.arange(n), -scores)) if scores[j] > 0][:k]
    recs = [{"i": cat[j], "w": cat[why[j][0]]} for j in order]
    for p in range(n):  # back-fill with popular items when neighbours run out
        if len(recs) >= k:
            break
        if p not in chosen and p not in order:
            recs.append({"i": cat[p], "w": None})
    out["itemknn"] = recs

    # ALS fold-in
    als = data["als"]
    f = np.array(als["factors"])
    gram = np.array(als["gram"])
    w = als["alpha"] * np.log1p(1.0)
    a = gram + w * f[chosen].T @ f[chosen] + als["reg"] * np.eye(f.shape[1])
    b = (1.0 + w) * f[chosen].sum(axis=0)
    p_vec = np.linalg.solve(a, b)
    sc = f @ p_vec
    sc[chosen] = -np.inf
    order = [j for j in np.lexsort((np.arange(n), -sc)) if np.isfinite(sc[j])][:k]
    unit = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    recs = []
    for j in order:
        sims = unit[chosen] @ unit[j]
        recs.append({"i": cat[j], "w": cat[chosen[int(np.argmax(sims))]]})
    out["als"] = recs
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    inter, names = load_clean(ROOT / "data" / "raw" / "online_retail_II.xlsx", ROOT / "data" / "cache")
    train, _ = temporal_split(inter, args.test_fraction)
    pop, knn, als = (m.fit(train) for m in build_models(args.seed))

    rng = np.random.default_rng(args.seed)
    sizes = train.groupby("user")["item"].nunique()
    candidates = sorted(sizes[(sizes >= 6) & (sizes <= 40)].index)
    chosen_users = [candidates[i] for i in sorted(rng.choice(len(candidates), N_CUSTOMERS, replace=False))]

    # item table: catalogue first (popularity order), then any other referenced items
    item_ids = list(pop.ranking[:N_CATALOGUE])
    pos = {it: i for i, it in enumerate(item_ids)}

    def idx(item):
        if item not in pos:
            pos[item] = len(item_ids)
            item_ids.append(item)
        return pos[item]

    customers = []
    for n, user in enumerate(chosen_users, start=1):
        recent = (
            train[train["user"] == user].sort_values(["ts", "item"], ascending=[False, True])
            .drop_duplicates("item")["item"].tolist()
        )
        entry = {
            "label": f"Sample customer {n}",
            "n_products": int(sizes[user]),
            "history": [idx(i) for i in recent[:HISTORY_SHOWN]],
            "recs": {},
        }
        for model in (pop, knn, als):
            recs = []
            for item in model.recommend(user, TOP_K):
                why = None if model is pop else model.explain(user, item)
                recs.append({"i": idx(item), "w": None if why is None else idx(why)})
            entry["recs"][model.name] = recs
        customers.append(entry)

    # picker data: neighbours and factors restricted to the catalogue
    cat_cols = np.array([knn.item_index[i] for i in pop.ranking[:N_CATALOGUE]])
    sim = knn.similarity[np.ix_(cat_cols, cat_cols)]
    knn_lists = []
    for row in sim:
        top = np.lexsort((np.arange(len(row)), -row))[:N_NEIGHBOURS]
        knn_lists.append([[int(j), round(float(row[j]), 4)] for j in top if row[j] > 0])
    factors = np.round(als.item_factors[cat_cols], 4)
    gram = np.round(als.gram, 4)

    data = {
        "meta": {
            "dataset": "UCI Online Retail II (CC BY 4.0)",
            "seed": args.seed,
            "note": "Independent re-implementation on public data. Illustrative recommendations only.",
        },
        "items": [[it, names.get(it, it)] for it in item_ids],
        "catalogue": list(range(N_CATALOGUE)),
        "knn": knn_lists,
        "als": {
            "alpha": als.alpha,
            "reg": als.reg,
            "factors": factors.tolist(),
            "gram": gram.tolist(),
        },
        "customers": customers,
    }

    # self-check baskets: reference outputs the page must reproduce exactly
    checks = []
    for basket in ([0, 5, 12], [3, 40, 41, 90], [7]):
        checks.append({"chosen": basket, "expected": demo_recommend(data, basket)})
    data["selfcheck"] = checks

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    text = json.dumps(data, separators=(",", ":"))
    (docs / "demo-data.json").write_text(text + "\n", encoding="utf-8")
    (docs / "demo-data.js").write_text("window.DEMO_DATA=" + text + ";\n", encoding="utf-8")
    print(f"items={len(item_ids)} customers={len(customers)} bytes={len(text):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
