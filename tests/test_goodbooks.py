"""Goodbooks-10k parsing and subsetting on tiny synthetic archives (the real data is never needed by the tests)."""
import zipfile

import pandas as pd

from ecomrec import registry
from ecomrec.data import goodbooks_interactions, goodbooks_item_info, read_goodbooks

RATINGS = "user_id,book_id,rating\n1,258,5\n2,4081,4\n2,260,5\n3,258,3\n1,260,4\n3,9296,2\n"
BOOKS = ('book_id,goodreads_book_id,authors,original_publication_year,title\n'
         '258,1,"A. Author, B. Author",1999.0,First Book (Series #1)\n'
         '260,2,C. Author,,"Second Book, The"\n'
         '4081,3,D. Author,2005.0,Third Book\n'
         '9296,4,E. Author,2010.0,Fourth Book\n')


def make_zips(tmp_path):
    ratings, books = tmp_path / "ratings.zip", tmp_path / "books.zip"
    with zipfile.ZipFile(ratings, "w") as archive:
        archive.writestr("ratings.csv", RATINGS)
    with zipfile.ZipFile(books, "w") as archive:
        archive.writestr("books.csv", BOOKS)
    return ratings, books


def big_ratings(n_users=60, n_books=30):
    rows = [(u, b, 1 + (u + b) % 5) for u in range(1, n_users + 1) for b in range(1, n_books + 1) if (u * b) % 3 != 0]
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"]).sample(frac=1.0, random_state=1).reset_index(drop=True)


def test_read_goodbooks_parses_both_archives(tmp_path):
    ratings, books = read_goodbooks(*make_zips(tmp_path))
    assert list(ratings.columns) == ["user_id", "book_id", "rating"] and len(ratings) == 6
    assert len(books) == 4 and books.loc[books["book_id"] == 260, "title"].iloc[0] == "Second Book, The"


def test_interactions_have_the_common_schema_and_keep_file_order_as_time(tmp_path):
    ratings, _ = read_goodbooks(*make_zips(tmp_path))
    inter = goodbooks_interactions(ratings)
    assert list(inter.columns) == ["user", "item", "ts", "basket"]
    assert inter["user"].map(type).eq(str).all() and inter["item"].map(type).eq(str).all()
    assert inter["ts"].is_monotonic_increasing and inter["ts"].is_unique  # position in the file stands in for time
    assert inter["item"].tolist() == ["258", "4081", "260", "258", "260", "9296"]
    assert (inter["basket"] == inter["user"]).all()


def test_every_rating_counts_as_an_interaction_whatever_the_score(tmp_path):
    ratings, _ = read_goodbooks(*make_zips(tmp_path))
    assert len(goodbooks_interactions(ratings)) == len(ratings)


def test_subset_samples_users_by_seed_and_keeps_their_rows_in_order():
    ratings = big_ratings()
    a = goodbooks_interactions(ratings, n_users=20, min_item_users=1, seed=3)
    b = goodbooks_interactions(ratings, n_users=20, min_item_users=1, seed=3)
    c = goodbooks_interactions(ratings, n_users=20, min_item_users=1, seed=4)
    assert a.equals(b) and a["user"].nunique() == 20 and set(a["user"]) != set(c["user"])
    assert a["ts"].is_monotonic_increasing
    full = goodbooks_interactions(ratings)
    kept = full[full["user"].isin(set(a["user"]))]
    assert kept["item"].tolist() == a["item"].tolist()  # same relative order as in the full file


def test_subset_drops_rarely_rated_items():
    ratings = big_ratings()
    sub = goodbooks_interactions(ratings, n_users=40, min_item_users=15, seed=0)
    assert (sub.groupby("item")["user"].nunique() >= 15).all()


def test_subset_larger_than_the_user_base_keeps_everyone():
    ratings = big_ratings(n_users=10)
    assert goodbooks_interactions(ratings, n_users=1000, min_item_users=1)["user"].nunique() == 10


def test_item_info_gives_titles(tmp_path):
    _, books = read_goodbooks(*make_zips(tmp_path))
    assert goodbooks_item_info(books)["258"] == "First Book (Series #1)"


def test_models_that_need_missing_columns_are_excluded():
    assert registry.excluded_models(has_price=True, has_time=True) == set()
    assert registry.excluded_models(has_price=False, has_time=True) == {"copurchase_price"}
    assert registry.excluded_models(has_price=False, has_time=False) == {"copurchase_price", "itemknn_recency"}


def test_build_models_honours_the_exclusion_set():
    names = [m.name for m in registry.build_models({}, seed=0, exclude={"itemknn_recency", "copurchase_price"})]
    assert "itemknn_recency" not in names and "copurchase_price" not in names and "ease" in names

