# Demo check (Playwright), 2026-10-14

`docs/index.html` was served locally with `python -m http.server` and driven with Playwright (Chromium) at two viewport widths.

| Check | 390 x 844 | 1280 x 800 |
|---|---|---|
| Page loads, title "Product Recommender Demo" | pass | pass |
| Console errors and uncaught page errors (fresh listener, reload, full interaction) | 0 | 0 |
| Horizontal overflow (`scrollWidth > innerWidth`) | none | none |
| Three model cards, 10 recommendations each, for a sample customer | pass | pass |
| Switching sample customer updates history and recommendations | pass | pass |
| Picker: search "lantern" returns matches, adding a product, example basket, remove and clear | pass | pass |
| Cards stack in one column at 390 px, sit in three columns at 1280 px | stacked | 3 columns |
| "Because you bought..." line present on item-CF and ALS recommendations | pass | pass |
| Browser JavaScript reproduces the Python reference output (`selfcheck`, 3 baskets x 3 models) | 9 of 9 identical | 9 of 9 identical |

The page shows no accuracy metrics. Screenshots: `docs/img/`.
