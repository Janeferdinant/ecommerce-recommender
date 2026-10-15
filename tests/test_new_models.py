"""EASE, BPR, recency-weighted item CF and the co-purchase baseline with business rules."""
import numpy as np
import pandas as pd
import pytest

from ecomrec.bpr import BPRModel, sample_negatives
from ecomrec.evaluate import evaluate
from ecomrec.linear import EASEModel
from ecomrec.models import ItemKNNModel, PopularityModel
from ecomrec.neighbours import CoPurchaseModel, RecencyItemKNNModel

K = 10


def toy_frame(rows):
    df = pd.DataFrame(rows, columns=["user", "item", "ts", "price"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def ndcg(split, model):
    return evaluate(split, seed=0, k=K, models=[model])["models"][model.name][f"ndcg@{K}"]


# ---------------------------------------------------------------- EASE
def test_ease_matches_the_closed_form(split):
    train, _ = split
    model = EASEModel(reg=25.0).fit(train)
    x = (model.counts > 0).toarray().astype(np.float64)
    p = np.linalg.inv(x.T @ x + 25.0 * np.eye(x.shape[1]))
    expected = -p / np.diag(p)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(model.weights, expected, atol=1e-6)


def test_ease_weights_have_a_zero_diagonal(split):
    train, _ = split
    model = EASEModel(reg=10.0).fit(train)
    assert np.all(np.diag(model.weights) == 0.0)


def test_ease_closed_form_minimises_the_regularised_loss(split):
    train, _ = split
    reg = 10.0
    model = EASEModel(reg=reg).fit(train)
    x = (model.counts > 0).toarray().astype(np.float64)

    def loss(b):
        return np.sum((x - x @ b) ** 2) + reg * np.sum(b**2)

    rng = np.random.default_rng(0)
    for _ in range(3):
        noise = rng.normal(0.0, 0.01, model.weights.shape)
        np.fill_diagonal(noise, 0.0)
        assert loss(model.weights + noise) > loss(model.weights)


def test_ease_recommends_new_items_only_and_is_deterministic(split):
    train, _ = split
    a, b = EASEModel(reg=10.0).fit(train), EASEModel(reg=10.0).fit(train)
    bought = train.groupby("user")["item"].apply(set).to_dict()
    for user, seen in bought.items():
        recs = a.recommend(user, k=K)
        assert not set(recs) & seen
        assert recs == b.recommend(user, k=K)


def test_ease_beats_popularity_on_planted_taste_groups(split):
    assert ndcg(split, EASEModel(reg=10.0)) > ndcg(split, PopularityModel())


def test_ease_rejects_a_non_positive_penalty():
    with pytest.raises(ValueError):
        EASEModel(reg=0.0)


# ---------------------------------------------------------------- BPR
def test_negative_sampling_never_returns_a_positive_item():
    n_items = 6
    positives = {0: {0, 1, 2, 3, 4}, 1: {1, 2, 3, 4, 5}}  # exactly one negative per user
    keys = np.array(sorted(u * n_items + i for u, items in positives.items() for i in items))
    users = np.array([0, 1] * 50)
    negatives = sample_negatives(np.random.default_rng(0), users, n_items, keys)
    assert set(negatives[users == 0]) == {5} and set(negatives[users == 1]) == {0}


def test_bpr_loss_goes_down_during_training(split):
    train, _ = split
    model = BPRModel(factors=8, epochs=8, lr=0.05, reg=0.001, seed=1).fit(train)
    assert len(model.loss_history) == 8
    assert model.loss_history[-1] < model.loss_history[0]


def test_bpr_is_deterministic_for_a_seed_and_changes_with_it(split):
    train, _ = split
    a = BPRModel(factors=8, epochs=3, seed=1).fit(train)
    b = BPRModel(factors=8, epochs=3, seed=1).fit(train)
    c = BPRModel(factors=8, epochs=3, seed=2).fit(train)
    np.testing.assert_array_equal(a.item_factors, b.item_factors)
    assert not np.allclose(a.item_factors, c.item_factors)


def test_bpr_recommends_new_items_only(split):
    train, _ = split
    model = BPRModel(factors=8, epochs=3, seed=1).fit(train)
    bought = train.groupby("user")["item"].apply(set).to_dict()
    for user, seen in list(bought.items())[:20]:
        recs = model.recommend(user, k=K)
        assert len(recs) == len(set(recs)) and not set(recs) & seen


def test_bpr_beats_popularity_on_planted_taste_groups(split):
    model = BPRModel(factors=8, epochs=25, lr=0.05, reg=0.001, seed=1)
    assert ndcg(split, model) > ndcg(split, PopularityModel())


def test_bpr_basket_mode_excludes_the_basket(split):
    train, _ = split
    model = BPRModel(factors=8, epochs=3, seed=1).fit(train)
    basket = model.items[:3]
    recs = model.recommend_for_items(basket, k=K)
    assert not set(recs) & set(basket) and len(recs) == K


# ---------------------------------------------------------------- recency-weighted item CF
def test_without_decay_the_recency_model_equals_item_cf(split):
    train, _ = split
    plain = ItemKNNModel().fit(train)
    recent = RecencyItemKNNModel(half_life_days=None).fit(train)
    for user in sorted(train["user"].unique())[:25]:
        assert recent.recommend(user, k=K) == plain.recommend(user, k=K)


def test_recent_purchases_outweigh_old_ones():
    rows = []
    for u in ("o1", "o2", "o3"):  # the old crowd buys A with X
        rows += [(u, "A", "2020-01-01", 1.0), (u, "X", "2020-01-01", 1.0)]
    for u in ("n1", "n2", "n3"):  # the recent crowd buys B with Y
        rows += [(u, "B", "2020-12-01", 1.0), (u, "Y", "2020-12-01", 1.0)]
    rows += [("t", "A", "2020-01-02", 1.0), ("t", "B", "2020-12-02", 1.0)]
    train = toy_frame(rows)
    assert RecencyItemKNNModel(half_life_days=30).fit(train).recommend("t", k=1) == ["Y"]
    assert RecencyItemKNNModel(half_life_days=30).fit(train).explain("t", "Y") == "B"


def test_recency_rejects_a_non_positive_half_life():
    with pytest.raises(ValueError):
        RecencyItemKNNModel(half_life_days=0.0)


# ---------------------------------------------------------------- co-purchase with business rules
def basket_log():
    rows = [
        ("u1", "A", "2020-01-01 10:00", 5.0), ("u1", "B", "2020-01-01 10:00", 1.0),
        ("u2", "A", "2020-01-02 10:00", 5.0), ("u2", "B", "2020-01-02 10:00", 1.0),
        ("u2", "A", "2020-02-02 10:00", 5.0), ("u2", "C", "2020-02-02 10:00", 20.0),
        ("u3", "A", "2020-03-02 10:00", 5.0), ("u3", "C", "2020-03-02 10:00", 20.0),
        ("t", "A", "2020-04-01 10:00", 5.0),
    ]
    return toy_frame(rows)


def test_copurchase_counts_baskets_not_customers():
    model = CoPurchaseModel(beta=0.0, min_support=1).fit(basket_log())
    assert model.cooccurrence(("A", "B")) == 2
    assert model.cooccurrence(("A", "C")) == 2


def test_minimum_support_rule_drops_rare_pairs():
    train = basket_log()
    loose = CoPurchaseModel(beta=0.0, min_support=2).fit(train)
    strict = CoPurchaseModel(beta=0.0, min_support=3).fit(train)
    assert loose.recommend("t", k=1)[0] in {"B", "C"}
    assert strict.scores_for_items(["A"]).max() == 0.0


def test_price_rule_lifts_the_expensive_item():
    train = basket_log()
    flat = CoPurchaseModel(beta=0.0, min_support=1, gamma=0.0).fit(train)
    priced = CoPurchaseModel(beta=0.0, min_support=1, gamma=1.0).fit(train)
    assert priced.recommend("t", k=2)[0] == "C"
    assert flat.recommend("t", k=2)[0] == "B"  # equal scores and equal popularity: item order decides


def test_model_names_reflect_the_rules():
    assert CoPurchaseModel(gamma=0.0).name == "copurchase"
    assert CoPurchaseModel(gamma=0.5).name == "copurchase_price"


def test_price_rule_needs_prices():
    train = basket_log().drop(columns=["price"])
    with pytest.raises(ValueError):
        CoPurchaseModel(gamma=0.5).fit(train)
    CoPurchaseModel(gamma=0.0).fit(train)  # fine without the rule


def test_copurchase_never_recommends_owned_items_and_beats_popularity(split):
    train, _ = split
    model = CoPurchaseModel(beta=0.5, min_support=1, gamma=0.0).fit(train)
    bought = train.groupby("user")["item"].apply(set).to_dict()
    for user, seen in bought.items():
        assert not set(model.recommend(user, k=K)) & seen
    assert ndcg(split, CoPurchaseModel(beta=0.5, min_support=1)) > ndcg(split, PopularityModel())


def test_explain_top_returns_shares_that_sum_to_one(split):
    train, _ = split
    for model in (ItemKNNModel(), EASEModel(reg=10.0), CoPurchaseModel(min_support=1)):
        model.fit(train)
        user = train["user"].iloc[0]
        item = model.recommend(user, k=1)[0]
        parts = model.explain_top(user, item, n=3)
        assert 1 <= len(parts) <= 3
        assert all(src in model.seen_items(user) for src, _ in parts)
        assert abs(sum(share for _, share in parts) - 1.0) < 1e-9  # shares are normalised over the parts shown
