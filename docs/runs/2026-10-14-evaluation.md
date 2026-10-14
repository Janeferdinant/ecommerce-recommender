# Evaluation run 2026-10-14

Status: Verified (run 2026-10-14). Independent re-implementation on public data
(UCI Online Retail II, CC BY 4.0). These are offline ranking metrics, not business impact.

- Started: 2026-10-14 04:00 UTC
- Command: `python scripts/run_evaluation.py --seed 42 --date 2026-10-14`
- Code version: 9ac97d3
- Environment: Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, CPU only, Windows
- Seed: 42 (ALS initialisation; no other randomness in the pipeline)
- Wall time: data loaded from the local cache of the cleaned table (the uncached read and clean of the workbook took 61 s on this machine in an earlier run of the same code); 19 s to fit and evaluate all models
- Customers: all eligible customers (no subsampling needed)

## Data after cleaning

| Quantity | Value |
|---|---|
| Purchase records (customer, product, basket) | 766,079 |
| Customers | 5,852 |
| Products | 4,619 |
| Span | 2009-12-01 to 2011-12-09 |

## Split

Temporal, by interaction count: the earliest 80% of purchase records are train
(strictly before 2011-09-12 12:27:00), the rest are test.

| Set | Records | Customers | Products |
|---|---|---|---|
| Train | 612,817 | 5,267 | 4,452 |
| Test | 153,262 | | |

Evaluation customers: 2,191 (seen in training and bought at least one product in the test period
that they had not bought in training). Recommendations exclude everything the customer bought in training; the
ground truth is only the new products bought in the test period.

## Results (k = 10, mean over evaluation customers; percentage = relative to the popularity baseline)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity baseline | 0.0407 | 0.0167 | 0.0434 |
| Item-based CF (cosine) | 0.0599 (+47%) | 0.0307 (+84%) | 0.0682 (+57%) |
| Implicit ALS (matrix factorisation) | 0.0628 (+54%) | 0.0329 (+97%) | 0.0713 (+64%) |

ALS hyperparameters (fixed, not tuned on the test period): {'factors': 32, 'iterations': 15, 'reg': 0.1, 'alpha': 10.0}.
