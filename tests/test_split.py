import pandas as pd
import pytest

from ecomrec.split import temporal_split


def test_temporal_split_has_no_leakage(split):
    train, test = split
    assert len(train) > 0 and len(test) > 0
    assert train.ts.max() <= test.ts.min()
    assert not set(zip(test.user, test.item, test.ts)) & set(zip(train.user, train.item, train.ts))


def test_split_is_strictly_before_cutoff(split):
    """Interactions sharing one timestamp (one basket) must never straddle the cut."""
    train, test = split
    assert train.ts.max() < test.ts.min()


def test_split_covers_every_interaction_once(interactions, split):
    train, test = split
    assert len(train) + len(test) == len(interactions)


def test_test_fraction_is_approximately_respected(interactions, split):
    _, test = split
    assert 0.1 <= len(test) / len(interactions) <= 0.35


def test_split_rejects_bad_fraction(interactions):
    with pytest.raises(ValueError):
        temporal_split(interactions, test_fraction=0.0)
    with pytest.raises(ValueError):
        temporal_split(interactions, test_fraction=1.0)


def test_split_does_not_depend_on_row_order(interactions):
    shuffled = interactions.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a_train, a_test = temporal_split(interactions, test_fraction=0.2)
    b_train, b_test = temporal_split(shuffled, test_fraction=0.2)
    key = lambda d: sorted(zip(d.user, d.item, d.ts))
    assert key(a_train) == key(b_train)
    assert key(a_test) == key(b_test)
