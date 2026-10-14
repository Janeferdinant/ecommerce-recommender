import numpy as np
import pytest

from ecomrec.evaluate import evaluate
from ecomrec.models import ALSModel, ItemKNNModel, PopularityModel

K = 10


def fitted_models(train, seed=7):
    return [
        PopularityModel().fit(train),
        ItemKNNModel().fit(train),
        ALSModel(factors=8, iterations=5, seed=seed).fit(train),
    ]


def test_recommendations_never_include_items_already_bought_in_training(split):
    train, _ = split
    bought = train.groupby("user")["item"].apply(set).to_dict()
    for model in fitted_models(train):
        for user, seen in bought.items():
            recs = model.recommend(user, k=K)
            assert len(recs) == len(set(recs)), f"{model.name}: duplicate items"
            assert not set(recs) & seen, f"{model.name} recommended an already-bought item to {user}"


def test_recommendations_are_at_most_k_and_ranked_lists(split):
    train, _ = split
    user = train.user.iloc[0]
    for model in fitted_models(train):
        assert 0 < len(model.recommend(user, k=5)) <= 5


def test_results_are_deterministic(split):
    assert evaluate(split, seed=7, als_params={"factors": 8, "iterations": 5}) == evaluate(
        split, seed=7, als_params={"factors": 8, "iterations": 5}
    )


def test_different_seeds_change_the_als_factors(split):
    train, _ = split
    a = ALSModel(factors=8, iterations=3, seed=1).fit(train)
    b = ALSModel(factors=8, iterations=3, seed=2).fit(train)
    assert not np.allclose(a.item_factors, b.item_factors)


def test_unknown_user_falls_back_to_popularity(split):
    train, _ = split
    pop = PopularityModel().fit(train)
    for model in fitted_models(train):
        assert model.recommend("never-seen-customer", k=K) == pop.recommend("never-seen-customer", k=K)


def test_popularity_ranks_by_distinct_buyers(split):
    train, _ = split
    counts = train.groupby("item")["user"].nunique().sort_values(ascending=False)
    pop = PopularityModel().fit(train)
    top = pop.ranking[:3]
    assert counts[top[0]] >= counts[top[1]] >= counts[top[2]]


def test_item_knn_recommends_same_taste_group_more_than_chance(split):
    """On the planted-group data a CF model must beat random group membership (25%)."""
    train, test = split
    model = ItemKNNModel().fit(train)
    hits = total = 0
    for user in train.user.unique()[:30]:
        group = int(user[1:]) % 4
        for item in model.recommend(user, k=5):
            hits += int(item[1:]) % 4 == group
            total += 1
    assert hits / total > 0.5


def test_explanations_point_to_items_the_user_bought(split):
    train, _ = split
    bought = train.groupby("user")["item"].apply(set).to_dict()
    for model in fitted_models(train)[1:]:
        user = next(iter(bought))
        for item in model.recommend(user, k=5):
            assert model.explain(user, item) in bought[user]


def test_als_fold_in_reproduces_trained_user_factor(split):
    """The demo computes picker recommendations by folding in; it must match training."""
    train, _ = split
    model = ALSModel(factors=8, iterations=4, seed=3).fit(train)
    user = train.user.iloc[0]
    row = model.user_index[user]
    items = model.seen_items(user)
    counts = [model.count(user, i) for i in items]
    folded = model.fold_in([model.item_index[i] for i in items], counts)
    assert np.allclose(folded, model.user_factors[row], atol=1e-8)
