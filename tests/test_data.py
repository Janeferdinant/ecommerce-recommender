import pandas as pd

from ecomrec.data import clean_transactions, item_names, to_interactions


def raw_frame():
    rows = [
        # Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
        ("500001", "85123A", "WHITE HANGING HEART", 6, "2010-12-01 08:26", 2.55, 17850.0, "United Kingdom"),
        ("500001", "85123A", "WHITE HANGING HEART", 2, "2010-12-01 08:26", 2.55, 17850.0, "United Kingdom"),
        ("500001", "71053", "WHITE METAL LANTERN", 6, "2010-12-01 08:26", 3.39, 17850.0, "United Kingdom"),
        ("C500002", "85123A", "WHITE HANGING HEART", -1, "2010-12-02 09:00", 2.55, 17850.0, "United Kingdom"),
        ("500003", "POST", "POSTAGE", 1, "2010-12-03 10:00", 18.0, 17850.0, "United Kingdom"),
        ("500004", "22423", "REGENCY CAKESTAND", 1, "2010-12-03 11:00", 12.75, None, "United Kingdom"),
        ("500005", "22423", "REGENCY CAKESTAND", 0, "2010-12-03 12:00", 12.75, 12583.0, "France"),
        ("500006", "22423", "REGENCY CAKESTAND", 1, "2010-12-03 13:00", 0.0, 12583.0, "France"),
        ("500007", "22423", "REGENCY CAKESTAND", 1, "2010-12-04 13:00", 12.75, 12583.0, "France"),
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    return pd.DataFrame(rows, columns=cols)


def test_cleaning_drops_cancellations_fees_anonymous_and_invalid_rows():
    clean = clean_transactions(raw_frame())
    assert set(clean.Invoice) == {"500001", "500007"}
    assert "POST" not in set(clean.StockCode)
    assert clean["Customer ID"].notna().all()
    assert (clean.Quantity > 0).all() and (clean.Price > 0).all()


def test_interactions_have_one_row_per_customer_item_basket():
    inter = to_interactions(clean_transactions(raw_frame()))
    assert list(inter.columns) == ["user", "item", "ts"]
    assert len(inter) == 3  # 85123A (two lines, one basket), 71053, and 22423
    assert set(inter.user) == {"17850", "12583"}


def test_item_names_pick_the_most_common_description():
    clean = clean_transactions(raw_frame())
    assert item_names(clean)["85123A"] == "White Hanging Heart"
