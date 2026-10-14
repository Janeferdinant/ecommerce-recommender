# Evaluation run 2026-10-14 (multi-seed, with engagement and revenue proxies)

Status: Verified (run 2026-10-14). Independent re-implementation on public data (UCI Online Retail II, CC BY 4.0).

**These are offline proxies computed on historical purchases. They are not the deployed engagement and revenue
gains.** The 15% engagement and 10% revenue figures on the CV are Reported results from an earlier deployment and
are not reproduced here.

- Started: 2026-10-14 04:36 UTC
- Command: `python scripts/run_evaluation.py --seeds 42 43 44 45 46`
- Code version: e4821db
- Environment: Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, CPU only, Windows
- Seeds: [42, 43, 44, 45, 46] (the seed only changes the ALS initialisation; popularity and item-based CF are deterministic)
- Wall time: 1 s to load the cleaned table (the first uncached read of the workbook takes about 80 s); 139 s to fit and evaluate all models for 5 seeds. Total well under the 20-minute CPU limit.
- ALS hyperparameters (selected on a validation window, see the tuning log): {'factors': 64, 'iterations': 15, 'reg': 1.0, 'alpha': 10.0}
- Customers: all eligible customers

## Data and split

| Quantity | Value |
|---|---|
| Purchase records (customer, product, basket) | 766,079 |
| Customers / products | 5,852 / 4,619 |
| Span | 2009-12-01 to 2011-12-09 |
| Train records (strictly before 2011-09-12 12:27:00) | 612,817 (5,267 customers, 4,452 products) |
| Test records | 153,262 |
| Evaluation customers | 2,191 |

Evaluation customers were seen in training and bought at least one product in the test period that they had not
bought in training. Recommendations exclude everything bought in training; ground truth is those new products.

## Metrics at k = 10 (mean ± sample sd over 5 seeds)

Revenue@10 is the mean, per evaluation customer, of the unit-price-weighted value (in GBP, as recorded in the
data) of the recommended products that the customer really bought in the test period. Prices are each product's
median unit price in the training period. Hit rate@10 is the share of customers with at least one correct
recommendation. Coverage@10 is the share of the 4,452-product training catalogue that appears in
anyone's top 10.

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit rate@10 | Coverage@10 | Revenue@10 (GBP) |
|---|---|---|---|---|---|---|
| Popularity baseline | 0.0407 ± 0.0000 | 0.0167 ± 0.0000 | 0.0434 ± 0.0000 | 0.2880 ± 0.0000 | 0.0180 ± 0.0000 | 1.4875 ± 0.0000 |
| Item-based CF (cosine) | 0.0599 ± 0.0000 | 0.0307 ± 0.0000 | 0.0682 ± 0.0000 | 0.3560 ± 0.0000 | 0.1768 ± 0.0000 | 1.4625 ± 0.0000 |
| Implicit ALS (matrix factorisation) | 0.0645 ± 0.0007 | 0.0340 ± 0.0006 | 0.0726 ± 0.0006 | 0.3953 ± 0.0032 | 0.3066 ± 0.0020 | 1.7042 ± 0.0275 |

## Relative uplift over the popularity baseline (95% paired bootstrap CI over customers, 1000 resamples)

| Model | Metric | Uplift | 95% interval |
|---|---|---|---|
| Item-based CF (cosine) | precision@10 | +47.4% | +36.5% to +59.7% |
| Item-based CF (cosine) | recall@10 | +83.7% | +60.1% to +110.7% |
| Item-based CF (cosine) | ndcg@10 | +57.1% | +44.0% to +71.1% |
| Item-based CF (cosine) | hit_rate@10 | +23.6% | +16.2% to +32.1% |
| Item-based CF (cosine) | revenue@10 | -1.7% | -10.9% to +7.8% |
| Item-based CF (cosine) | coverage@10 | +884% | not bootstrapped (catalogue-level) |
| Implicit ALS (matrix factorisation) | precision@10 | +58.7% | +46.4% to +72.6% |
| Implicit ALS (matrix factorisation) | recall@10 | +103.2% | +73.7% to +136.7% |
| Implicit ALS (matrix factorisation) | ndcg@10 | +67.1% | +52.0% to +83.1% |
| Implicit ALS (matrix factorisation) | hit_rate@10 | +37.2% | +28.2% to +47.9% |
| Implicit ALS (matrix factorisation) | revenue@10 | +14.6% | +3.5% to +27.5% |
| Implicit ALS (matrix factorisation) | coverage@10 | +1607% | not bootstrapped (catalogue-level) |

Seed-to-seed variation is the sd in the table above; the intervals reflect which customers happened to be in the
test period. A single temporal split is used, so neither captures variation across time windows.
