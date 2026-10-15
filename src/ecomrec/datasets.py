"""The datasets of the study and the evaluation protocol that goes with each."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import load_clean
from .split import split_by_protocol, validation_split_by_protocol

DATASETS = ("retail",)


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
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
