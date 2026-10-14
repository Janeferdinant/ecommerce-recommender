"""Load and clean UCI Online Retail II into a (user, item, ts) purchase log."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
# Real products have a 5-digit code with an optional letter suffix (e.g. 85123A).
# Everything else (POST, D, M, BANK CHARGES, gift vouchers, ...) is a fee or an adjustment.
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"


def read_workbook(xlsx_path: Path) -> pd.DataFrame:
    """Read both sheets of the workbook with openpyxl in streaming mode (a few minutes)."""
    from openpyxl import load_workbook

    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    frames = []
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        header = list(next(rows))
        frames.append(pd.DataFrame(list(rows), columns=header))
    workbook.close()
    return pd.concat(frames, ignore_index=True)[RAW_COLUMNS]


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep completed purchases of real products by identified customers."""
    df = raw.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df = df[~df["Invoice"].str.upper().str.startswith("C")]  # cancellations / returns
    df = df[df["Customer ID"].notna()]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[df["StockCode"].str.match(PRODUCT_CODE, case=False)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.reset_index(drop=True)


def to_interactions(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer, product, basket): columns user, item, ts."""
    out = pd.DataFrame(
        {
            "user": clean["Customer ID"].astype(float).astype(int).astype(str),
            "item": clean["StockCode"],
            "ts": clean["InvoiceDate"],
            "invoice": clean["Invoice"],
        }
    )
    out = out.drop_duplicates(["user", "item", "invoice"])
    return out[["user", "item", "ts"]].reset_index(drop=True)


def item_names(clean: pd.DataFrame) -> dict:
    """Most frequent (title-cased) description per product code."""
    desc = clean.dropna(subset=["Description"]).copy()
    desc["Description"] = desc["Description"].astype(str).str.strip()
    top = (
        desc.groupby(["StockCode", "Description"]).size().reset_index(name="n")
        .sort_values(["StockCode", "n", "Description"], ascending=[True, False, True])
        .drop_duplicates("StockCode")
    )
    return {row.StockCode: row.Description.title() for row in top.itertuples()}


def load_clean(xlsx_path: Path, cache_dir: Path):
    """Return ``(interactions, names)``, caching the cleaned tables as CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    inter_path, names_path = cache_dir / "interactions.csv", cache_dir / "item_names.csv"
    if inter_path.exists() and names_path.exists():
        inter = pd.read_csv(inter_path, dtype={"user": str, "item": str}, parse_dates=["ts"])
        names = pd.read_csv(names_path, dtype=str).set_index("item")["name"].to_dict()
        return inter, names
    clean = clean_transactions(read_workbook(xlsx_path))
    inter, names = to_interactions(clean), item_names(clean)
    inter.to_csv(inter_path, index=False)
    pd.Series(names, name="name").rename_axis("item").reset_index().to_csv(names_path, index=False)
    return inter, names
