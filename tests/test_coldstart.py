"""New-customer (cold start) protocol: hide all but the first n purchases of a customer sample."""
import numpy as np
import pandas as pd

from ecomrec.coldstart import cold_start_eval, truncate_histories
from ecomrec.models import ItemKNNModel, PopularityModel


def test_truncation_keeps_the_first_n_distinct_items_of_chosen_customers_only():
    rows = [("a", "i1", "2020-01-01"), ("a", "i2", "2020-01-02"), ("a", "i3", "2020-01-03"), ("a", "i1", "2020-01-04"),
            ("b", "i1", "2020-01-01"), ("b", "i4", "2020-01-02")]
    df = pd.DataFrame(rows, columns=["user", "item", "ts"]).assign(ts=lambda d: pd.to_datetime(d["ts"]))
    out = truncate_histories(df, users=["a"], n=2)
    assert sorted(out.loc[out["user"] == "a", "item"].unique()) == ["i1", "i2"]
    assert len(out.loc[out["user"] == "b"]) == 2  # untouched
    out_all = truncate_histories(df, users=["a", "b"], n=5)
    assert len(out_all) == len(df)  # fewer purchases than n: nothing removed


def test_truncation_never_uses_later_purchases(interactions):
    users = sorted(interactions["user"].unique())[:10]
    out = truncate_histories(interactions, users, n=2)
    for user in users:
        kept = out[out["user"] == user]
        full = interactions[interactions["user"] == user].sort_values(["ts", "item"])
        first_two = full.drop_duplicates("item")["item"].tolist()[:2]
        assert set(kept["item"]) == set(first_two)


def test_cold_start_models_see_less_data_than_the_full_fit(split):
    train, test = split
    seen_sizes = []

    class Spy(PopularityModel):
        def fit(self, data):
            seen_sizes.append(len(data))
            return super().fit(data)

    result = cold_start_eval((train, test), {"spy": Spy}, n_values=(1, 2), k=5, seed=0, cold_fraction=0.5)
    assert seen_sizes[0] == len(train)  # full-history reference fit first
    assert all(size < len(train) for size in seen_sizes[1:]) and len(seen_sizes) == 3
    assert result["n_cold_users"] > 0


def test_cold_start_result_structure_and_ranges(split):
    result = cold_start_eval(split, {"pop": PopularityModel, "knn": ItemKNNModel}, n_values=(1, 3), k=5, seed=1,
                             cold_fraction=0.5)
    for name in ("pop", "knn"):
        for key in ("full", 1, 3):
            hits = result["per_user"][name][key]["hit_rate@5"]
            assert len(hits) == result["n_cold_users"] and set(np.unique(hits)) <= {0.0, 1.0}
            assert 0.0 <= result["summary"][name][key]["ndcg@5"] <= 1.0


def test_cold_start_is_deterministic(split):
    a = cold_start_eval(split, {"knn": ItemKNNModel}, n_values=(2,), k=5, seed=4, cold_fraction=0.5)
    b = cold_start_eval(split, {"knn": ItemKNNModel}, n_values=(2,), k=5, seed=4, cold_fraction=0.5)
    assert a["users"] == b["users"] and a["summary"] == b["summary"]


def test_recommendations_exclude_the_visible_items(split):
    train, test = split
    recorded = {}

    class Spy(ItemKNNModel):
        def recommend_for_items(self, items, k=10):
            recs = super().recommend_for_items(items, k)
            recorded[tuple(items)] = recs
            return recs

    cold_start_eval((train, test), {"spy": Spy}, n_values=(2,), k=5, seed=0, cold_fraction=0.5)
    assert recorded and all(not set(items) & set(recs) for items, recs in recorded.items())
