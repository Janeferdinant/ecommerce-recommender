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

## Python libraries (installed by the user, not redistributed here)

| Library | License | Use |
|---|---|---|
| NumPy | BSD-3-Clause | numerical arrays, ALS solver |
| pandas | BSD-3-Clause | data cleaning and splits |
| SciPy | BSD-3-Clause | sparse matrices |
| openpyxl | MIT | reading the dataset workbook |
| pytest | MIT | tests |
| ruff | MIT | linting (development only) |

No third-party source code is copied into this repository. The algorithms follow published papers (see the README, "Credits & citation"); the implementations are original.
