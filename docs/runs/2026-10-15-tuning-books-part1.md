# Tuning run 2026-10-15: Goodbooks-10k (subset) (copurchase, ease, bpr)

Hyperparameters were chosen on a validation window inside the training period. The test period was not used.

- Started: 2026-10-15 17:06 UTC
- Command: `python scripts/tune_models.py --dataset books --models itemknn_recency copurchase ease bpr`
- Protocol: temporal split on file order (the ratings carry no timestamps): the earliest 80% of the subset's ratings train, the latest 20% test
- Fit window: 555,679 records; validation window: 138,920 records (4,376 customers)
- Seed 42 (only ALS and BPR use it); k = 10
- Environment: Python 3.11.9, Windows AMD64, 16 logical CPUs, CPU only
- Selection rule: highest validation NDCG@10, earliest grid point wins ties (price-aware co-purchase: validation Revenue@10)
- Validation NDCG@10 of the untuned baselines: popularity 0.0388, item-based CF 0.0531

## Co-purchase (basket co-occurrence)

Objective: validation ndcg@10. Time: 410 s.

| beta | min_support | NDCG@10 | Hit rate@10 |  |
|---|---|---|---|---|
| 0.0 | 1 | 0.0480 | 0.1995 |  |
| 0.0 | 2 | 0.0482 | 0.2007 |  |
| 0.0 | 3 | 0.0481 | 0.2007 |  |
| 0.0 | 5 | 0.0480 | 0.2001 |  |
| 0.25 | 1 | 0.0520 | 0.2221 |  |
| 0.25 | 2 | 0.0519 | 0.2224 |  |
| 0.25 | 3 | 0.0517 | 0.2212 |  |
| 0.25 | 5 | 0.0516 | 0.2181 |  |
| 0.5 | 1 | 0.0572 | 0.2429 |  |
| 0.5 | 2 | 0.0572 | 0.2447 | **selected** |
| 0.5 | 3 | 0.0567 | 0.2444 |  |
| 0.5 | 5 | 0.0565 | 0.2408 |  |
| 0.75 | 1 | 0.0518 | 0.2420 |  |
| 0.75 | 2 | 0.0531 | 0.2438 |  |
| 0.75 | 3 | 0.0537 | 0.2395 |  |
| 0.75 | 5 | 0.0544 | 0.2444 |  |

Selected: `{'beta': 0.5, 'min_support': 2}`

## EASE

Objective: validation ndcg@10. Time: 168 s.

| reg | NDCG@10 | Hit rate@10 |  |
|---|---|---|---|
| 10.0 | 0.0580 | 0.3123 |  |
| 30.0 | 0.0626 | 0.3223 |  |
| 100.0 | 0.0645 | 0.3266 |  |
| 300.0 | 0.0647 | 0.3272 | **selected** |
| 1000.0 | 0.0628 | 0.3159 |  |
| 3000.0 | 0.0583 | 0.2924 |  |
| 10000.0 | 0.0520 | 0.2511 |  |

Selected: `{'reg': 300.0}`

## BPR matrix factorisation

Objective: validation ndcg@10. Time: 403 s.

| factors | lr | reg | NDCG@10 | Hit rate@10 |  |
|---|---|---|---|---|---|
| 32 | 0.02 | 0.001 | 0.0504 | 0.2276 |  |
| 32 | 0.02 | 0.01 | 0.0527 | 0.2401 |  |
| 32 | 0.05 | 0.001 | 0.0546 | 0.2704 |  |
| 32 | 0.05 | 0.01 | 0.0586 | 0.2851 |  |
| 64 | 0.02 | 0.001 | 0.0496 | 0.2282 |  |
| 64 | 0.02 | 0.01 | 0.0507 | 0.2343 |  |
| 64 | 0.05 | 0.001 | 0.0560 | 0.2817 |  |
| 64 | 0.05 | 0.01 | 0.0607 | 0.2985 | **selected** |

Selected: `{'factors': 64, 'lr': 0.05, 'reg': 0.01}`

Grids are coarse and the differences between the better settings are small; read the selections as reasonable defaults, not sharp optima.
Settings at the edge of a grid are flagged in the README.
