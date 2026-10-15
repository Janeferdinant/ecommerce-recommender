# Demo check 2026-10-15: shop and books, "why was this recommended" panel

Checked with Playwright (Chromium) against `docs/` served from a local static server (`python -m http.server --bind 127.0.0.1 --directory docs`); no deployment.

| Check | Result |
|---|---|
| Console messages at 1280 px and 390 px (errors and warnings, whole session) | 0 |
| Horizontal overflow at 390 px and 1280 px | none (document width 375 px inside a 390 px viewport, scrollbar excluded) |
| Dataset switch (Online shop / Books); the Books data loads on demand from `demo-data-books.js` | works; the "Sample customer" tab is hidden for Books, which has no customer samples |
| Sample-customer mode (shop): eight model cards, each recommendation opens a "Why?" panel with up to three source items and shares | works |
| Picker mode with the preset baskets, both datasets; the books page shows six cards (no price rule, no recency model: the ratings have no timestamps) | works |
| JS against the Python reference (`ecomrec.demo_ref`), top-10 lists and top contributing item for every model | 56 of 56 identical (shop: 4 baskets x 8 models = 32; books: 4 baskets x 6 models = 24) |
| Metric terms (hit rate, NDCG, precision, recall, coverage, revenue@) anywhere in the page text | none |
| Attribution of the embedded books excerpt (CC BY-SA 4.0, Goodbooks-10k, Zygmunt Zając) in the page footer | present |

Data files: `docs/demo-data.json` and `docs/demo-data.js` (shop, 2.2 MB each), `docs/demo-data-books.js` (books, 1.3 MB). They hold titles and model-derived numbers only; no customer ids, user ids or ratings. The books file is an excerpt of CC BY-SA 4.0 material (see `THIRD_PARTY_NOTICES.md`). Screenshots: `docs/img/demo-desktop-customer.png`, `demo-desktop-books.png`, `demo-mobile-shop.png`, `demo-mobile-books.png`.
