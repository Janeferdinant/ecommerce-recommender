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


def leave_last_out(df: pd.DataFrame, n_test: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each customer's latest ``n_test`` interactions.

    Interactions are ordered by ``(ts, item)`` within a customer, so timestamp ties
    are broken by item id and the result does not depend on row order. Customers
    with ``n_test`` or fewer interactions keep everything in train.
    """
    if n_test < 1:
        raise ValueError("n_test must be at least 1")
    ordered = df.sort_values(["user", "ts", "item"], kind="mergesort")
    from_end = ordered.groupby("user").cumcount(ascending=False)
    sizes = ordered.groupby("user")["item"].transform("size")
    is_test = (from_end < n_test) & (sizes > n_test)
    test = ordered[is_test].sort_index().reset_index(drop=True)
    train = ordered[~is_test].sort_index().reset_index(drop=True)
    return train, test


def split_by_protocol(df: pd.DataFrame, protocol: str, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``"temporal"`` (global cut in time) or ``"leave_last"`` (each customer's last interaction)."""
    if protocol == "temporal":
        return temporal_split(df, test_fraction)
    if protocol == "leave_last":
        return leave_last_out(df)
    raise ValueError(f"unknown protocol {protocol!r}")


def validation_split_by_protocol(train: pd.DataFrame, protocol: str,
                                 val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same protocol again *inside* the training period; the test period is never used."""
    return split_by_protocol(train, protocol, val_fraction)
