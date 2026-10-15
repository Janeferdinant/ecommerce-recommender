# Study run 2026-10-15: Goodbooks-10k (subset)

Status: Verified (run 2026-10-15). Offline evaluation on historical data; not measured business impact.

- Started: 2026-10-15 17:23 UTC
- Command: `python scripts/run_study.py --dataset books --seeds 42 43 44 45 46`
- Code version: 7dd1106 (+ uncommitted changes)
- Environment: Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, Windows AMD64, 16 logical CPUs, CPU only
- Seeds: [42, 43, 44, 45, 46] (only ALS and BPR use the seed; the other models are deterministic and were run once)
- Wall time: 532 s for fitting and evaluating all models and seeds
- Protocol: temporal split on file order (the ratings carry no timestamps): the earliest 80% of the subset's ratings train, the latest 20% test
- Settings: `configs/tuned-books.json`, selected on a validation window inside the training period

## Data and split

| Quantity | Value |
|---|---|
| Interactions | 868,249 |
| Customers / items | 8,000 / 8,361 |
| Span | not available (no timestamps; file order is used) |
| Train / test interactions | 694,599 / 173,650 |
| Cut-off (temporal protocol only) | row position (no timestamps) |
| Evaluation customers | 4,092 |
| Training catalogue | 8,085 items |
| Relevant test items outside the training catalogue (unreachable) | 10.9% |

Evaluation customers were seen in training and have at least one test item that they had not interacted with in
training. Recommendations exclude everything the customer interacted with in training; the ground truth is those new
items.

## Tuned settings (validation window)

| Model | Settings | Validation NDCG@10 |
|---|---|---|
| Popularity | `{}` | 0.0388 |
| Item-based CF | `{}` | 0.0531 |
| Co-purchase (basket co-occurrence) | `{'beta': 0.5, 'min_support': 2}` | 0.0572 |
| Implicit ALS | `{'factors': 64, 'reg': 0.1, 'iterations': 10, 'alpha': 10.0}` | 0.0601 |
| EASE | `{'reg': 300.0}` | 0.0647 |
| BPR matrix factorisation | `{'factors': 64, 'lr': 0.05, 'reg': 0.01}` | 0.0607 |

## Accuracy at k = 10 (test window; mean ± sample sd over seeds where the model is seeded)

| Model | Hit rate@10 | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Popularity | 0.1879 | 0.0482 | 0.0467 | 0.0147 |
| Item-based CF | 0.2556 | 0.0615 | 0.0581 | 0.0203 |
| Co-purchase (basket co-occurrence) | 0.2832 | 0.0674 | 0.0628 | 0.0238 |
| Implicit ALS | 0.3374 ± 0.0040 | 0.0696 ± 0.0006 | 0.0658 ± 0.0004 | 0.0256 ± 0.0004 |
| EASE | 0.3573 | 0.0726 | 0.0689 | 0.0284 |
| BPR matrix factorisation | 0.3173 ± 0.0074 | 0.0667 ± 0.0014 | 0.0631 ± 0.0013 | 0.0244 ± 0.0004 |

## Beyond accuracy at k = 10

Coverage is the share of the training catalogue that appears in anyone's list. Novelty is the mean self-information
(-log2 of the share of customers who interacted with the item); higher means less popular items. Diversity is one minus
the mean pairwise cosine similarity (on customer-item interaction vectors from the training period) between the items in a
list; higher means more varied lists.

| Model | Coverage@10 | Novelty@10 (bits) | Diversity@10 |
|---|---|---|---|
| Popularity | 0.0066 | 1.7663 | 0.5120 |
| Item-based CF | 0.2275 | 2.8519 | 0.6161 |
| Co-purchase (basket co-occurrence) | 0.3692 | 3.8133 | 0.6544 |
| Implicit ALS | 0.2297 ± 0.0011 | 3.7374 ± 0.0060 | 0.7727 ± 0.0004 |
| EASE | 0.3054 | 3.6468 | 0.7786 |
| BPR matrix factorisation | 0.2216 ± 0.0043 | 3.7385 ± 0.0150 | 0.7469 ± 0.0013 |

## Uplift over popularity (95% paired bootstrap intervals over customers, 1000 resamples)

| Model | hit_rate@10 | ndcg@10 | precision@10 | recall@10 |
|---|---|---|---|---|
| Item-based CF | +36.0% (+28.6% to +44.2%) | +27.7% (+18.9% to +37.3%) | +24.5% (+16.1% to +33.2%) | +38.2% (+24.5% to +54.5%) |
| Co-purchase (basket co-occurrence) | +50.7% (+41.1% to +60.6%) | +39.8% (+28.5% to +52.5%) | +34.4% (+23.7% to +46.1%) | +62.2% (+43.4% to +83.8%) |
| Implicit ALS | +79.6% (+68.0% to +92.5%) | +44.6% (+34.1% to +58.2%) | +40.9% (+30.8% to +53.7%) | +74.5% (+54.9% to +99.6%) |
| EASE | +90.1% (+78.4% to +103.2%) | +50.7% (+39.5% to +64.4%) | +47.5% (+36.3% to +60.8%) | +93.3% (+69.6% to +120.1%) |
| BPR matrix factorisation | +68.8% (+58.7% to +80.2%) | +38.6% (+28.9% to +50.7%) | +35.2% (+26.0% to +46.6%) | +66.1% (+47.8% to +87.5%) |

## Best two models, compared (paired bootstrap over customers, 2000 resamples)

The two models with the highest validation NDCG@10 (chosen before looking at the test window) were
EASE (A) and BPR matrix factorisation (B). Validation ranking: EASE, BPR matrix factorisation, Implicit ALS, Co-purchase (basket co-occurrence), Item-based CF, Popularity.
Test ranking by NDCG@10: EASE, Implicit ALS, Co-purchase (basket co-occurrence), ...

| Metric | A | B | A - B (95% CI) | Relative (95% CI) | p (two-sided bootstrap) |
|---|---|---|---|---|---|
| ndcg@10 | 0.0726 | 0.0667 | +0.0058 (+0.0028 to +0.0090) | +8.7% (+4.0% to +13.7%) | < 0.0005 |
| hit_rate@10 | 0.3573 | 0.3173 | +0.0400 (+0.0281 to +0.0513) | +12.6% (+8.8% to +16.4%) | < 0.0005 |
| novelty@10 | 3.6468 | 3.7385 | -0.0917 (-0.1166 to -0.0676) | -2.5% (-3.1% to -1.8%) | < 0.0005 |
| diversity@10 | 0.7786 | 0.7469 | +0.0317 (+0.0292 to +0.0343) | +4.2% (+3.9% to +4.6%) | < 0.0005 |

## Popularity bias

Items are split into 10 equal-sized bins by training popularity (bin 1 = the best-selling tenth of the
catalogue). First table: share of all recommended slots per bin (%). Second table: share of the relevant (customer, item)
pairs of that bin that were found in the top 10 (%).

Share of recommended slots per popularity bin (%):

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Item-based CF | 87.9 | 4.5 | 2.8 | 1.5 | 0.9 | 0.8 | 0.5 | 0.5 | 0.4 | 0.3 |
| Co-purchase (basket co-occurrence) | 72.7 | 7.6 | 5.5 | 3.8 | 2.8 | 2.5 | 1.8 | 1.6 | 1.2 | 0.7 |
| Implicit ALS | 87.9 | 8.8 | 2.3 | 0.6 | 0.2 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| EASE | 84.8 | 7.8 | 3.4 | 1.7 | 0.9 | 0.6 | 0.3 | 0.3 | 0.1 | 0.1 |
| BPR matrix factorisation | 86.9 | 8.5 | 2.7 | 0.9 | 0.4 | 0.2 | 0.1 | 0.2 | 0.1 | 0.0 |

Recall of relevant items per popularity bin (%):

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | 5.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Item-based CF | 5.6 | 1.4 | 1.4 | 1.0 | 1.2 | 1.0 | 0.8 | 1.0 | 0.5 | 0.2 |
| Co-purchase (basket co-occurrence) | 5.0 | 1.9 | 2.1 | 2.1 | 2.6 | 2.2 | 1.9 | 2.1 | 1.2 | 0.4 |
| Implicit ALS | 7.0 | 2.1 | 1.1 | 0.5 | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| EASE | 6.7 | 2.4 | 1.7 | 1.2 | 1.2 | 0.6 | 0.4 | 0.6 | 0.2 | 0.1 |
| BPR matrix factorisation | 6.9 | 1.7 | 0.9 | 0.5 | 0.3 | 0.1 | 0.1 | 0.3 | 0.1 | 0.0 |

Seed-to-seed variation is the sd shown above; the intervals reflect which customers are in the test window. One
split is used, so neither captures variation across time windows.
