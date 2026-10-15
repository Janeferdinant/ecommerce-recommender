"""The datasets of the study and the evaluation protocol that goes with each."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import goodbooks_interactions, goodbooks_item_info, load_clean, read_goodbooks
from .registry import excluded_models
from .split import split_by_protocol, validation_split_by_protocol

DATASETS = ("retail", "books")

# Goodbooks-10k subset: a seeded random sample of users and the books that at least this many of them rated
BOOKS_N_USERS = 8000
BOOKS_MIN_ITEM_USERS = 20
BOOKS_SEED = 0


@dataclass
class Bundle:
    name: str
    label: str
    inter: pd.DataFrame
    names: dict[str, str]
    genres: dict[str, frozenset[str]] | None
    has_price: bool
    protocol: str  # "temporal" (global cut in time) or "leave_last" (each customer's last interaction)
    protocol_text: str
    has_time: bool = True

    @property
    def excluded_models(self) -> set[str]:
        return excluded_models(has_price=self.has_price, has_time=self.has_time)

    def split(self, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        return split_by_protocol(self.inter, self.protocol, test_fraction)

    def validation_split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return validation_split_by_protocol(train, self.protocol)


def load_bundle(name: str, root: Path) -> Bundle:
    """Load a dataset that ``scripts/download_data.py`` has already fetched and verified."""
    if name == "retail":
        inter, names = load_clean(root / "data" / "raw" / "online_retail_II.xlsx", root / "data" / "cache")
        return Bundle(
            "retail", "UCI Online Retail II", inter, names, None, True, "temporal",
            "temporal split: the earliest 80% of purchase records train, the latest 20% test",
        )
    if name == "books":
        ratings, books = read_goodbooks(root / "data" / "raw" / "goodbooks-ratings.zip",
                                        root / "data" / "raw" / "goodbooks-books.zip")
        inter = goodbooks_interactions(ratings, BOOKS_N_USERS, BOOKS_MIN_ITEM_USERS, BOOKS_SEED)
        return Bundle(
            "books", "Goodbooks-10k (subset)", inter, goodbooks_item_info(books), None, False, "temporal",
            "temporal split on file order (the ratings carry no timestamps): the earliest 80% of the subset's ratings "
            "train, the latest 20% test",
            has_time=False,
        )
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
