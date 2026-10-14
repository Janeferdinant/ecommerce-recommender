from ecomrec.tuning import tune_als, validation_split

GRID = {"factors": [4, 8], "reg": [0.1], "iterations": [2, 3]}


def test_validation_window_sits_inside_the_training_period(split):
    train, test = split
    fit, val = validation_split(train, 0.2)
    assert fit.ts.max() < val.ts.min()
    assert val.ts.max() <= train.ts.max() < test.ts.min()
    assert len(fit) + len(val) == len(train)


def test_tune_als_returns_a_grid_member_and_a_full_table(split):
    train, _ = split
    best, table = tune_als(train, GRID, seed=1)
    assert best["factors"] in GRID["factors"] and best["iterations"] in GRID["iterations"]
    assert len(table) == 4
    best_row = next(r for r in table if all(r[k] == best[k] for k in GRID))
    assert best_row["ndcg@10"] == max(row["ndcg@10"] for row in table)


def test_tuning_is_deterministic(split):
    train, _ = split
    assert tune_als(train, GRID, seed=1) == tune_als(train, GRID, seed=1)
