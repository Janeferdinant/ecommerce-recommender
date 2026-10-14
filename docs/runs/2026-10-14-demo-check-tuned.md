# Demo re-check (Playwright) after the tuned ALS model, 2026-10-14

`docs/index.html` was rebuilt from newly generated `demo-data.json` (tuned ALS: 64 factors, regularisation 1.0, 15 iterations),
served locally with `python -m http.server` and driven with Playwright (Chromium) at two viewport widths.

| Check | 390 x 844 | 1280 x 800 |
|---|---|---|
| Page loads, title "Product Recommender Demo" | pass | pass |
| Console errors, warnings and uncaught page errors (fresh listener, full interaction) | 0 | 0 |
| Horizontal overflow (`scrollWidth > innerWidth`) | none | none |
| Three model cards, 10 recommendations each, for a sample customer | pass | pass |
| Switching sample customer updates history and recommendations | pass | pass |
| Picker: search, add product, example basket, remove, clear | pass | pass |
| Layout | cards in 1 column | cards in 3 columns |
| Browser JavaScript reproduces the Python reference output (`selfcheck`, 3 baskets x 3 models) | 9 of 9 identical | 9 of 9 identical |
| Page text contains no metric terms (precision, recall, NDCG, hit rate, revenue, coverage, 15%, 10%) | pass | pass |

The page shows no accuracy, engagement or revenue metrics. Screenshots: `docs/img/`.
