# Tuning run 2026-10-15: Goodbooks-10k (subset) (als)

Hyperparameters were chosen on a validation window inside the training period. The test period was not used.

- Started: 2026-10-15 17:06 UTC
- Command: `python scripts/tune_models.py --dataset books --models als`
- Protocol: temporal split on file order (the ratings carry no timestamps): the earliest 80% of the subset's ratings train, the latest 20% test
- Fit window: 555,679 records; validation window: 138,920 records (4,376 customers)
- Seed 42 (only ALS and BPR use it); k = 10
- Environment: Python 3.11.9, Windows AMD64, 16 logical CPUs, CPU only
- Selection rule: highest validation NDCG@10, earliest grid point wins ties (price-aware co-purchase: validation Revenue@10)
- Validation NDCG@10 of the untuned baselines: popularity 0.0388, item-based CF 0.0531

## Implicit ALS

Objective: validation ndcg@10. Time: 601 s.

| factors | reg | iterations | NDCG@10 | Hit rate@10 |  |
|---|---|---|---|---|---|
| 16 | 0.01 | 10 | 0.0589 | 0.2896 |  |
| 16 | 0.01 | 15 | 0.0586 | 0.2857 |  |
| 16 | 0.1 | 10 | 0.0589 | 0.2899 |  |
| 16 | 0.1 | 15 | 0.0586 | 0.2857 |  |
| 16 | 1.0 | 10 | 0.0590 | 0.2906 |  |
| 16 | 1.0 | 15 | 0.0585 | 0.2857 |  |
| 32 | 0.01 | 10 | 0.0584 | 0.2881 |  |
| 32 | 0.01 | 15 | 0.0587 | 0.2866 |  |
| 32 | 0.1 | 10 | 0.0584 | 0.2881 |  |
| 32 | 0.1 | 15 | 0.0587 | 0.2863 |  |
| 32 | 1.0 | 10 | 0.0585 | 0.2884 |  |
| 32 | 1.0 | 15 | 0.0588 | 0.2866 |  |
| 64 | 0.01 | 10 | 0.0601 | 0.2994 |  |
| 64 | 0.01 | 15 | 0.0597 | 0.3013 |  |
| 64 | 0.1 | 10 | 0.0601 | 0.2994 | **selected** |
| 64 | 0.1 | 15 | 0.0597 | 0.3013 |  |
| 64 | 1.0 | 10 | 0.0600 | 0.2988 |  |
| 64 | 1.0 | 15 | 0.0599 | 0.3019 |  |

Selected: `{'factors': 64, 'reg': 0.1, 'iterations': 10, 'alpha': 10.0}`

Grids are coarse and the differences between the better settings are small; read the selections as reasonable defaults, not sharp optima.
Settings at the edge of a grid are flagged in the README.
