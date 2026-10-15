"""Model registry, generic validation-window tuning and the tuned-settings file."""
import pytest

from ecomrec import registry
from ecomrec.bpr import BPRModel
from ecomrec.linear import EASEModel
from ecomrec.models import ALSModel
from ecomrec.neighbours import CoPurchaseModel, RecencyItemKNNModel
from ecomrec.tuning import tune_model


def test_every_model_in_the_study_can_be_built():
    assert registry.MODEL_ORDER == [
        "popularity", "itemknn", "itemknn_recency", "copurchase", "copurchase_price", "als", "ease", "bpr",
    ]
    assert isinstance(registry.make_model("ease", {"reg": 10.0}, seed=0), EASEModel)
    assert isinstance(registry.make_model("bpr", {"factors": 4, "epochs": 1}, seed=3), BPRModel)
    assert isinstance(registry.make_model("als", {"factors": 4, "iterations": 1, "reg": 1.0}, seed=3), ALSModel)
    assert isinstance(registry.make_model("itemknn_recency", {"half_life_days": 30}, seed=0), RecencyItemKNNModel)
    assert registry.make_model("copurchase_price", {"beta": 0.5, "min_support": 2, "gamma": 0.5}, seed=0).name == "copurchase_price"
    with pytest.raises(KeyError):
        registry.make_model("transformer", {}, seed=0)


def test_seed_only_reaches_models_that_use_it():
    assert registry.make_model("bpr", {}, seed=11).seed == 11
    assert registry.make_model("als", {}, seed=11).seed == 11


def test_tuned_file_round_trips(tmp_path):
    tuned = {"dataset": "toy", "models": {"ease": {"params": {"reg": 100.0}, "val_ndcg@10": 0.1, "objective": "ndcg@10"}}}
    path = tmp_path / "tuned.json"
    registry.write_tuned(path, tuned)
    assert registry.read_tuned(path) == tuned


def test_build_models_uses_tuned_params_and_skips_price_models_without_prices():
    tuned = {"models": {name: {"params": dict(registry.GRIDS[name]["defaults"])} for name in registry.GRIDS}}
    with_price = registry.build_models(tuned, seed=0, has_price=True)
    assert [m.name for m in with_price] == registry.MODEL_ORDER
    without = registry.build_models(tuned, seed=0, has_price=False)
    assert "copurchase_price" not in [m.name for m in without]


def test_grids_cover_the_tunable_models_with_an_objective_each():
    for name, spec in registry.GRIDS.items():
        assert spec["grid"] and spec["objective"] in {"ndcg@10", "revenue@10"} and spec["defaults"]
        for key, values in spec["grid"].items():
            assert len(values) >= 2, (name, key)


def test_tune_model_picks_a_grid_member_using_only_the_training_period(split):
    train, test = split
    grid = {"reg": [1.0, 10.0, 100.0]}
    best, table = tune_model(lambda **p: EASEModel(**p), grid, train, protocol="temporal", objective="ndcg@10")
    assert best["reg"] in grid["reg"] and len(table) == 3
    assert max(row["ndcg@10"] for row in table) == next(r for r in table if r["reg"] == best["reg"])["ndcg@10"]
    assert all(row["n_val_users"] > 0 for row in table)


def test_tune_model_can_optimise_revenue(split):
    train, _ = split
    grid = {"gamma": [0.0, 1.0]}
    best, table = tune_model(lambda **p: CoPurchaseModel(min_support=1, **p), grid, train, protocol="temporal",
                             objective="revenue@10")
    assert best["gamma"] in (0.0, 1.0)
    assert max(r["revenue@10"] for r in table) == next(r for r in table if r["gamma"] == best["gamma"])["revenue@10"]


def test_tune_model_is_deterministic_and_breaks_ties_towards_the_first_grid_point(split):
    train, _ = split
    grid = {"reg": [10.0, 10.0]}
    a = tune_model(lambda **p: EASEModel(**p), grid, train, protocol="temporal", objective="ndcg@10")
    assert a == tune_model(lambda **p: EASEModel(**p), grid, train, protocol="temporal", objective="ndcg@10")
    assert a[0] == {"reg": 10.0}


def test_leave_last_protocol_is_supported(split):
    train, _ = split
    best, table = tune_model(lambda **p: EASEModel(**p), {"reg": [1.0, 50.0]}, train, protocol="leave_last",
                             objective="ndcg@10")
    assert best["reg"] in (1.0, 50.0) and len(table) == 2
