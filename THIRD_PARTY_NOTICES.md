# Third-party notices

The original code, tests and documentation in this repository are © Jane Rose Ferdinant, licensed CC BY-NC-ND 4.0 (see `LICENSE`). The components below keep their own licenses.

## Online Retail II dataset

- **Source:** https://archive.ics.uci.edu/dataset/502/online+retail+ii (DOI 10.24432/C5CG6D)
- **Creator:** Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **How it is used here:**
  - The raw workbook is **not** included in this repository. `scripts/download_data.py` fetches it from the UCI site and verifies a recorded SHA-256 checksum.
  - `docs/demo-data.json` and `docs/demo-data.js` contain material derived from the dataset: product codes, product descriptions, item-item similarity scores and model factors computed from the purchase records. Customer identifiers are not included; sample customers are anonymous labels.
  - Changes made: transactions were cleaned (cancellations, fees, rows without a customer or with non-positive quantity or price removed), aggregated into customer-product-basket records and used to train recommender models.
- The CC BY 4.0 license of the dataset applies to the dataset and to material derived from it. It is not replaced by the license of this repository's original work.

## Goodbooks-10k dataset

- **Source:** https://github.com/zygmuntz/goodbooks-10k (release v1.0, assets `ratings.zip` and `books.zip`, published 2017-09-19; downloaded at run time from https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/)
- **Creator:** Zygmunt Zając (GitHub: zygmuntz). The ratings and book metadata were collected from Goodreads (goodreads.com), which is the original source of the underlying content.
- **License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).** Evidence, checked on 2026-10-15: the repository's `LICENSE` file (https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/LICENSE, added in commit b4bb02b7b8, "CSV samples, license", 2017-09-13) contains exactly this text:

  > This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
  >
  > To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

  License deed: https://creativecommons.org/licenses/by-sa/4.0/. The repository's README (https://github.com/zygmuntz/goodbooks-10k) describes the data (six million ratings of ten thousand books, with book metadata and tags) and does not add terms. I rely on the dataset author's stated licence; I have not checked whether Goodreads' own terms would restrict the original content, which is a residual risk that cannot be resolved from the dataset's licence alone.
- **How it is used here:**
  - The release archives are **not** included in this repository. `scripts/download_data.py --dataset books` downloads them and verifies recorded SHA-256 checksums (ratings.zip `2dd0ea8e...ebd2e`, books.zip `1e5e29f4...f7bd4f`).
  - Changes made: a seeded random sample of 8,000 users (and the books rated by at least 20 of them) was used; every rating, whatever its value, counts as one implicit interaction; the ratings carry no timestamps, so the position of a row in the file (which the dataset's documentation says is sorted by time) stands in for time.
  - **Embedded excerpt.** `docs/demo-data-books.js` contains an excerpt derived from the dataset: 600 book titles (from `books.csv`) and numbers computed from the ratings by the models (item-item scores and factors). It contains no ratings, no user ids and no row order. **This excerpt is licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), separately from the CC BY-NC-ND 4.0 that covers the original code and text of this repository.** Reuse of the excerpt must credit Zygmunt Zając and the Goodbooks-10k dataset, link the licence, state that changes were made (selection of books, derived model numbers), and share any adaptation under CC BY-SA 4.0. Reuse of the excerpt under CC BY-SA does not extend to the repository's code or documentation.
- The licence of the dataset applies to the dataset and to material derived from it. It is not replaced by the licence of this repository's original work.

## Python libraries (installed by the user, not redistributed here)

| Library | License | Use |
|---|---|---|
| NumPy | BSD-3-Clause | numerical arrays, ALS solver |
| pandas | BSD-3-Clause | data cleaning and splits |
| SciPy | BSD-3-Clause | sparse matrices |
| openpyxl | MIT | reading the dataset workbook |
| pytest | MIT | tests |
| ruff | MIT | linting (development only) |
| mypy | MIT | type checking (development only) |
| pandas-stubs | BSD-3-Clause | type stubs (development only) |
| matplotlib | Matplotlib license (PSF-based) | README figures (development only) |

No third-party source code is copied into this repository. The algorithms follow published papers (see the README, "Credits & citation"); the implementations are original.
