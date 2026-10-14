"""Shared fixtures: a small synthetic purchase log with planted taste groups.

The tests never need the real dataset, so they run in seconds on any machine.
"""
import numpy as np
import pandas as pd
import pytest

from ecomrec.split import temporal_split


def make_interactions(n_users=60, n_items=40, n_groups=4, seed=0):
    """Customers belong to a taste group and mostly buy items from that group."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2011-01-01")
    rows = []
    for u in range(n_users):
        group = u % n_groups
        group_items = [i for i in range(n_items) if i % n_groups == group]
        other_items = [i for i in range(n_items) if i % n_groups != group]
        for order in range(int(rng.integers(3, 8))):
            ts = start + pd.Timedelta(days=int(rng.integers(0, 300)), minutes=int(rng.integers(0, 600)))
            basket = set(rng.choice(group_items, size=int(rng.integers(2, 5)), replace=False).tolist())
            if rng.random() < 0.3:
                basket.add(int(rng.choice(other_items)))
            for item in basket:
                rows.append((f"u{u}", f"i{item}", ts))
    df = pd.DataFrame(rows, columns=["user", "item", "ts"])
    return df.drop_duplicates().reset_index(drop=True)


@pytest.fixture(scope="session")
def interactions():
    return make_interactions()


@pytest.fixture(scope="session")
def split(interactions):
    return temporal_split(interactions, test_fraction=0.2)
