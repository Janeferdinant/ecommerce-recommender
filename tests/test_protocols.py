"""Split protocols: temporal (retail) and per-customer leave-last-out."""
import pandas as pd
import pytest

from ecomrec.split import leave_last_out, split_by_protocol, validation_split_by_protocol


def log():
    rows = [
        ("a", "i1", "2020-01-01 10:00"), ("a", "i2", "2020-01-02 10:00"), ("a", "i3", "2020-01-03 10:00"),
        ("b", "i2", "2020-01-01 09:00"), ("b", "i4", "2020-01-05 09:00"),
        ("c", "i1", "2020-01-04 09:00"),  # a single interaction: nothing to hold out
    ]
    return pd.DataFrame(rows, columns=["user", "item", "ts"]).assign(ts=lambda d: pd.to_datetime(d["ts"]))


def test_leave_last_out_holds_out_each_customers_latest_item():
    train, test = leave_last_out(log())
    assert dict(zip(test["user"], test["item"], strict=True)) == {"a": "i3", "b": "i4"}
    assert len(train) + len(test) == len(log())
    assert not set(map(tuple, train[["user", "item"]].values)) & set(map(tuple, test[["user", "item"]].values))


def test_customers_with_a_single_interaction_stay_in_train_only():
    train, test = leave_last_out(log())
    assert "c" in set(train["user"]) and "c" not in set(test["user"])


def test_leave_last_out_does_not_depend_on_row_order():
    df = log()
    a_train, a_test = leave_last_out(df)
    b_train, b_test = leave_last_out(df.sample(frac=1.0, random_state=3))
    key = ["user", "item"]
    assert a_test.sort_values(key).reset_index(drop=True)[key].equals(b_test.sort_values(key).reset_index(drop=True)[key])
    assert len(a_train) == len(b_train)


def test_timestamp_ties_are_broken_by_item_id():
    df = pd.DataFrame({"user": ["u", "u", "u"], "item": ["z", "a", "m"], "ts": pd.to_datetime(["2020-01-01"] * 3)})
    _, test = leave_last_out(df)
    assert test["item"].tolist() == ["z"]  # latest by (ts, item)


def test_n_test_greater_than_one():
    train, test = leave_last_out(log(), n_test=2)
    assert sorted(test.loc[test["user"] == "a", "item"]) == ["i2", "i3"]
    assert "b" not in set(test["user"])  # two interactions only: needs more than n_test to hold any out


def test_split_by_protocol_dispatches_and_rejects_unknown(interactions):
    t_train, t_test = split_by_protocol(interactions, "temporal", 0.2)
    assert t_train["ts"].max() < t_test["ts"].min()
    l_train, l_test = split_by_protocol(interactions, "leave_last")
    assert l_test.groupby("user").size().max() == 1
    with pytest.raises(ValueError):
        split_by_protocol(interactions, "random")


def test_validation_split_stays_inside_the_training_period(split):
    train, test = split
    for protocol in ("temporal", "leave_last"):
        fit, val = validation_split_by_protocol(train, protocol)
        assert len(fit) + len(val) == len(train)
        assert val["ts"].max() <= train["ts"].max() < test["ts"].min()
