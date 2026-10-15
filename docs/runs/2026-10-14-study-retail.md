# Study run 2026-10-14: UCI Online Retail II

Status: Verified (run 2026-10-14). Offline evaluation on historical data; not measured business impact.

- Started: 2026-10-14 21:48 UTC
- Command: `python scripts/run_study.py --dataset retail --seeds 42 43 44 45 46`
- Code version: fd35ad3
- Environment: Python 3.11.9, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, Windows AMD64, 16 logical CPUs, CPU only
- Seeds: [42, 43, 44, 45, 46] (only ALS and BPR use the seed; the other models are deterministic and were run once)
- Wall time: 294 s for fitting and evaluating all models and seeds
- Protocol: temporal split: the earliest 80% of purchase records train, the latest 20% test
- Settings: `configs/tuned-retail.json`, selected on a validation window inside the training period

## Data and split

| Quantity | Value |
|---|---|
| Interactions | 766,079 |
| Customers / items | 5,852 / 4,619 |
| Span | 2009-12-01 to 2011-12-09 |
| Train / test interactions | 612,817 / 153,262 |
| Cut-off (temporal protocol only) | 2011-09-12 12:27:00 |
| Evaluation customers | 2,191 |
| Training catalogue | 4,452 items |
| Relevant test items outside the training catalogue (unreachable) | 11.6% |

Evaluation customers were seen in training and have at least one test item that they had not interacted with in
training. Recommendations exclude everything the customer interacted with in training; the ground truth is those new
items.

## Tuned settings (validation window)

| Model | Settings | Validation NDCG@10 |
|---|---|---|
| Popularity | `{}` | 0.0524 |
| Item-based CF | `{}` | 0.0888 |
| Item-based CF, recency-weighted | `{'half_life_days': 30}` | 0.1006 |
| Co-purchase (basket co-occurrence) | `{'beta': 0.5, 'min_support': 5}` | 0.0812 |
| Co-purchase + price-aware re-ranking | `{'beta': 0.5, 'min_support': 5, 'gamma': 1.5}` | 0.0332 |
| Implicit ALS | `{'factors': 64, 'reg': 1.0, 'iterations': 15, 'alpha': 10.0}` | 0.0756 |
| EASE | `{'reg': 1000.0}` | 0.1030 |
| BPR matrix factorisation | `{'factors': 64, 'lr': 0.02, 'reg': 0.01}` | 0.0695 |

## Accuracy at k = 10 (test window; mean ± sample sd over seeds where the model is seeded)

| Model | Hit rate@10 | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Popularity | 0.2880 | 0.0434 | 0.0407 | 0.0167 |
| Item-based CF | 0.3560 | 0.0682 | 0.0599 | 0.0307 |
| Item-based CF, recency-weighted | 0.3756 | 0.0799 | 0.0668 | 0.0333 |
| Co-purchase (basket co-occurrence) | 0.3651 | 0.0679 | 0.0604 | 0.0320 |
| Co-purchase + price-aware re-ranking | 0.1356 | 0.0204 | 0.0170 | 0.0094 |
| Implicit ALS | 0.3953 ± 0.0032 | 0.0726 ± 0.0006 | 0.0645 ± 0.0007 | 0.0340 ± 0.0006 |
| EASE | 0.4614 | 0.0901 | 0.0782 | 0.0421 |
| BPR matrix factorisation | 0.3585 ± 0.0063 | 0.0614 ± 0.0029 | 0.0564 ± 0.0026 | 0.0234 ± 0.0013 |

## Beyond accuracy at k = 10

Coverage is the share of the training catalogue that appears in anyone's list. Novelty is the mean self-information
(-log2 of the share of customers who interacted with the item); higher means less popular items. Diversity is one minus
the mean pairwise cosine similarity (on customer-item interaction vectors from the training period) between the items in a
list; higher means more varied lists.

| Model | Coverage@10 | Novelty@10 (bits) | Diversity@10 |
|---|---|---|---|
| Popularity | 0.0180 | 2.6173 | 0.6962 |
| Item-based CF | 0.1768 | 3.1841 | 0.6638 |
| Item-based CF, recency-weighted | 0.3333 | 4.5394 | 0.7286 |
| Co-purchase (basket co-occurrence) | 0.4149 | 3.6677 | 0.7250 |
| Co-purchase + price-aware re-ranking | 0.1076 | 4.5427 | 0.8474 |
| Implicit ALS | 0.3066 ± 0.0020 | 3.9198 ± 0.0032 | 0.7814 ± 0.0004 |
| EASE | 0.2457 | 3.6054 | 0.7504 |
| BPR matrix factorisation | 0.1543 ± 0.0041 | 3.1328 ± 0.0119 | 0.7188 ± 0.0014 |

## Revenue proxy (retail only)

| Model | Revenue@10 (GBP/customer) | Share of the oracle's revenue | Revenue with test-period prices (check) |
|---|---|---|---|
| Popularity | 1.4875 | 0.0364 | 1.4930 |
| Item-based CF | 1.4625 | 0.0358 | 1.4844 |
| Item-based CF, recency-weighted | 1.6053 | 0.0393 | 1.6148 |
| Co-purchase (basket co-occurrence) | 1.3733 | 0.0336 | 1.3840 |
| Co-purchase + price-aware re-ranking | 2.1544 | 0.0528 | 2.1676 |
| Implicit ALS | 1.7042 ± 0.0275 | 0.0417 ± 0.0007 | 1.7264 ± 0.0275 |
| EASE | 2.0693 | 0.0507 | 2.1061 |
| BPR matrix factorisation | 1.6915 ± 0.0745 | 0.0414 ± 0.0018 | 1.7088 ± 0.0743 |

Median ratio of test-period to training-period unit price over the relevant items: 1.000. The oracle picks the 10 most valuable of each customer's relevant products. Revenue@k sums training-period median unit prices of the recommended products the customer really bought in the test period; it ignores quantity and margin and cannot capture sales a recommendation causes.

## Uplift over popularity (95% paired bootstrap intervals over customers, 1000 resamples)

| Model | hit_rate@10 | ndcg@10 | precision@10 | recall@10 | revenue@10 |
|---|---|---|---|---|---|
| Item-based CF | +23.6% (+16.2% to +32.1%) | +57.1% (+44.0% to +71.1%) | +47.4% (+36.5% to +59.7%) | +83.7% (+60.1% to +110.7%) | -1.7% (-10.9% to +7.8%) |
| Item-based CF, recency-weighted | +30.4% (+21.8% to +42.1%) | +84.0% (+67.7% to +103.7%) | +64.2% (+50.4% to +79.4%) | +99.2% (+70.2% to +133.8%) | +7.9% (-3.2% to +21.1%) |
| Co-purchase (basket co-occurrence) | +26.8% (+18.9% to +36.5%) | +56.2% (+43.0% to +70.5%) | +48.5% (+37.2% to +61.1%) | +91.3% (+64.4% to +122.3%) | -7.7% (-17.6% to +3.0%) |
| Co-purchase + price-aware re-ranking | -52.9% (-58.1% to -47.6%) | -52.9% (-58.5% to -46.8%) | -58.2% (-63.3% to -53.0%) | -44.1% (-57.9% to -25.8%) | +44.8% (+28.1% to +63.8%) |
| Implicit ALS | +37.2% (+28.2% to +47.9%) | +67.1% (+52.0% to +83.1%) | +58.7% (+46.4% to +72.6%) | +103.2% (+73.7% to +136.7%) | +14.6% (+3.5% to +27.5%) |
| EASE | +60.2% (+50.6% to +73.3%) | +107.5% (+90.5% to +127.1%) | +92.4% (+78.1% to +109.3%) | +151.3% (+117.4% to +191.2%) | +39.1% (+26.6% to +53.6%) |
| BPR matrix factorisation | +24.5% (+18.4% to +31.9%) | +41.5% (+32.6% to +51.4%) | +38.6% (+30.6% to +47.8%) | +39.8% (+26.4% to +55.8%) | +13.7% (+6.6% to +21.8%) |

## Best two models, compared (paired bootstrap over customers, 2000 resamples)

The two models with the highest validation NDCG@10 (chosen before looking at the test window) were
EASE (A) and Item-based CF, recency-weighted (B). Validation ranking: EASE, Item-based CF, recency-weighted, Item-based CF, Co-purchase (basket co-occurrence), Implicit ALS, BPR matrix factorisation, Popularity, Co-purchase + price-aware re-ranking.
Test ranking by NDCG@10: EASE, Item-based CF, recency-weighted, Implicit ALS, ...

| Metric | A | B | A - B (95% CI) | Relative (95% CI) | p (two-sided bootstrap) |
|---|---|---|---|---|---|
| ndcg@10 | 0.0901 | 0.0799 | +0.0102 (+0.0047 to +0.0157) | +12.8% (+5.6% to +20.5%) | 0.0010 |
| hit_rate@10 | 0.4614 | 0.3756 | +0.0858 (+0.0634 to +0.1082) | +22.8% (+16.4% to +29.8%) | < 0.0005 |
| revenue@10 | 2.0693 | 1.6053 | +0.4640 (+0.3280 to +0.6114) | +28.9% (+19.4% to +39.6%) | < 0.0005 |
| novelty@10 | 3.6054 | 4.5394 | -0.9341 (-0.9874 to -0.8813) | -20.6% (-21.5% to -19.6%) | < 0.0005 |
| diversity@10 | 0.7504 | 0.7286 | +0.0218 (+0.0180 to +0.0257) | +3.0% (+2.5% to +3.5%) | < 0.0005 |

## Popularity bias

Items are split into 10 equal-sized bins by training popularity (bin 1 = the best-selling tenth of the
catalogue). First table: share of all recommended slots per bin (%). Second table: share of the relevant (customer, item)
pairs of that bin that were found in the top 10 (%).

Share of recommended slots per popularity bin (%):

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Item-based CF | 95.9 | 2.3 | 0.7 | 0.3 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 |
| Item-based CF, recency-weighted | 63.3 | 13.7 | 8.8 | 3.2 | 2.5 | 1.4 | 1.9 | 0.9 | 1.9 | 2.3 |
| Co-purchase (basket co-occurrence) | 80.8 | 6.8 | 4.2 | 2.8 | 1.6 | 1.3 | 1.0 | 1.0 | 0.5 | 0.0 |
| Co-purchase + price-aware re-ranking | 56.4 | 15.5 | 15.4 | 2.2 | 2.6 | 6.0 | 1.3 | 0.6 | 0.2 | 0.0 |
| Implicit ALS | 78.1 | 17.0 | 3.9 | 0.8 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| EASE | 89.2 | 8.1 | 1.9 | 0.6 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| BPR matrix factorisation | 96.8 | 2.7 | 0.4 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Recall of relevant items per popularity bin (%):

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | 5.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Item-based CF | 7.9 | 0.4 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Item-based CF, recency-weighted | 6.6 | 2.2 | 1.2 | 0.9 | 0.5 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 |
| Co-purchase (basket co-occurrence) | 7.1 | 0.8 | 0.5 | 0.4 | 0.3 | 0.3 | 0.4 | 0.2 | 0.1 | 0.0 |
| Co-purchase + price-aware re-ranking | 1.6 | 0.7 | 0.3 | 0.1 | 0.1 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| Implicit ALS | 7.3 | 1.9 | 0.5 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| EASE | 9.8 | 1.2 | 0.3 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BPR matrix factorisation | 7.4 | 0.4 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Seed-to-seed variation is the sd shown above; the intervals reflect which customers are in the test window. One
split is used, so neither captures variation across time windows.
