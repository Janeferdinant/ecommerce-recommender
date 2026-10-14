"""Temporal train/test split."""
import pandas as pd


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split interactions in time: train is strictly earlier than test.

    ``df`` needs the columns ``user``, ``item`` and ``ts``. The cutoff is the
    timestamp at the ``1 - test_fraction`` quantile of interactions. Train keeps
    ``ts < cutoff`` and test keeps ``ts >= cutoff``, so rows that share one
    timestamp (one basket) always land on the same side. The result does not
    depend on row order and uses no randomness.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    ordered = df.sort_values("ts", kind="mergesort")
    cutoff = ordered["ts"].iloc[int(round(len(ordered) * (1.0 - test_fraction)))]
    train = df[df["ts"] < cutoff].reset_index(drop=True)
    test = df[df["ts"] >= cutoff].reset_index(drop=True)
    return train, test
