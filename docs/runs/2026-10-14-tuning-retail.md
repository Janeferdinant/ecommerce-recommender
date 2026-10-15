# Tuning run 2026-10-14: UCI Online Retail II (itemknn_recency, copurchase, copurchase_price, als, ease, bpr)

Hyperparameters were chosen on a validation window inside the training period. The test period was not used.

- Started: 2026-10-14 21:32 UTC
- Command: `python scripts/tune_models.py --dataset retail --models itemknn_recency copurchase copurchase_price als ease bpr`
- Protocol: temporal split: the earliest 80% of purchase records train, the latest 20% test
- Fit window: 490,156 records; validation window: 122,661 records (2,454 customers)
- Seed 42 (only ALS and BPR use it); k = 10
- Environment: Python 3.11.9, Windows AMD64, 16 logical CPUs, CPU only
- Selection rule: highest validation NDCG@10, earliest grid point wins ties (price-aware co-purchase: validation Revenue@10)
- Validation NDCG@10 of the untuned baselines: popularity 0.0524, item-based CF 0.0888

## Item-based CF, recency-weighted

Objective: validation ndcg@10. Time: 71 s.

| half_life_days | NDCG@10 | Hit rate@10 | Revenue@10 |  |
|---|---|---|---|---|
| 7 | 0.0859 | 0.3631 | 1.7227 |  |
| 30 | 0.1006 | 0.4710 | 2.0662 | **selected** |
| 90 | 0.0974 | 0.4561 | 1.9955 |  |
| 180 | 0.0940 | 0.4444 | 1.9248 |  |
| 365 | 0.0920 | 0.4264 | 1.9058 |  |
| none | 0.0888 | 0.4296 | 1.9153 |  |

Selected: `{'half_life_days': 30}`

## Co-purchase (basket co-occurrence)

Objective: validation ndcg@10. Time: 234 s.

| beta | min_support | NDCG@10 | Hit rate@10 | Revenue@10 |  |
|---|---|---|---|---|---|
| 0.0 | 1 | 0.0703 | 0.3870 | 1.8008 |  |
| 0.0 | 2 | 0.0706 | 0.3870 | 1.8112 |  |
| 0.0 | 3 | 0.0701 | 0.3860 | 1.7889 |  |
| 0.0 | 5 | 0.0699 | 0.3881 | 1.7873 |  |
| 0.25 | 1 | 0.0786 | 0.4152 | 1.8205 |  |
| 0.25 | 2 | 0.0787 | 0.4125 | 1.8212 |  |
| 0.25 | 3 | 0.0782 | 0.4120 | 1.8163 |  |
| 0.25 | 5 | 0.0776 | 0.4104 | 1.7950 |  |
| 0.5 | 1 | 0.0773 | 0.3913 | 1.5825 |  |
| 0.5 | 2 | 0.0797 | 0.4062 | 1.6536 |  |
| 0.5 | 3 | 0.0806 | 0.4067 | 1.6823 |  |
| 0.5 | 5 | 0.0812 | 0.4115 | 1.7380 | **selected** |
| 0.75 | 1 | 0.0355 | 0.2052 | 0.7243 |  |
| 0.75 | 2 | 0.0530 | 0.2865 | 1.0676 |  |
| 0.75 | 3 | 0.0589 | 0.3137 | 1.1789 |  |
| 0.75 | 5 | 0.0646 | 0.3493 | 1.3209 |  |

Selected: `{'beta': 0.5, 'min_support': 5}`

## Co-purchase + price-aware re-ranking

Objective: validation revenue@10. Time: 68 s.

| gamma | NDCG@10 | Hit rate@10 | Revenue@10 |  |
|---|---|---|---|---|
| 0.0 | 0.0812 | 0.4115 | 1.7380 |  |
| 0.25 | 0.0701 | 0.3833 | 2.8206 |  |
| 0.5 | 0.0555 | 0.3163 | 3.4982 |  |
| 1.0 | 0.0422 | 0.2573 | 3.6518 |  |
| 1.5 | 0.0332 | 0.2164 | 3.6796 | **selected** |

Selected: `{'beta': 0.5, 'min_support': 5, 'gamma': 1.5}`

## Implicit ALS

Objective: validation ndcg@10. Time: 328 s.

| factors | reg | iterations | NDCG@10 | Hit rate@10 | Revenue@10 |  |
|---|---|---|---|---|---|---|
| 16 | 0.01 | 10 | 0.0728 | 0.3907 | 1.8437 |  |
| 16 | 0.01 | 15 | 0.0725 | 0.3939 | 1.8428 |  |
| 16 | 0.1 | 10 | 0.0728 | 0.3907 | 1.8485 |  |
| 16 | 0.1 | 15 | 0.0726 | 0.3939 | 1.8428 |  |
| 16 | 1.0 | 10 | 0.0730 | 0.3918 | 1.8557 |  |
| 16 | 1.0 | 15 | 0.0726 | 0.3945 | 1.8519 |  |
| 32 | 0.01 | 10 | 0.0750 | 0.4019 | 2.0506 |  |
| 32 | 0.01 | 15 | 0.0751 | 0.4099 | 2.0756 |  |
| 32 | 0.1 | 10 | 0.0750 | 0.4019 | 2.0506 |  |
| 32 | 0.1 | 15 | 0.0750 | 0.4088 | 2.0757 |  |
| 32 | 1.0 | 10 | 0.0751 | 0.4014 | 2.0482 |  |
| 32 | 1.0 | 15 | 0.0752 | 0.4083 | 2.0846 |  |
| 64 | 0.01 | 10 | 0.0744 | 0.4094 | 1.9428 |  |
| 64 | 0.01 | 15 | 0.0755 | 0.4184 | 1.9899 |  |
| 64 | 0.1 | 10 | 0.0744 | 0.4094 | 1.9417 |  |
| 64 | 0.1 | 15 | 0.0754 | 0.4189 | 1.9883 |  |
| 64 | 1.0 | 10 | 0.0746 | 0.4104 | 1.9440 |  |
| 64 | 1.0 | 15 | 0.0756 | 0.4184 | 1.9900 | **selected** |

Selected: `{'factors': 64, 'reg': 1.0, 'iterations': 15, 'alpha': 10.0}`

## EASE

Objective: validation ndcg@10. Time: 45 s.

| reg | NDCG@10 | Hit rate@10 | Revenue@10 |  |
|---|---|---|---|---|
| 10.0 | 0.0842 | 0.4423 | 2.0779 |  |
| 30.0 | 0.0907 | 0.4689 | 2.2607 |  |
| 100.0 | 0.0957 | 0.4886 | 2.3433 |  |
| 300.0 | 0.1011 | 0.4997 | 2.4741 |  |
| 1000.0 | 0.1030 | 0.4997 | 2.5065 | **selected** |
| 3000.0 | 0.0997 | 0.4859 | 2.4300 |  |
| 10000.0 | 0.0907 | 0.4641 | 2.3294 |  |

Selected: `{'reg': 1000.0}`

## BPR matrix factorisation

Objective: validation ndcg@10. Time: 130 s.

| factors | lr | reg | NDCG@10 | Hit rate@10 | Revenue@10 |  |
|---|---|---|---|---|---|---|
| 32 | 0.02 | 0.001 | 0.0638 | 0.3711 | 1.8517 |  |
| 32 | 0.02 | 0.01 | 0.0642 | 0.3748 | 1.8817 |  |
| 32 | 0.05 | 0.001 | 0.0607 | 0.3557 | 1.6165 |  |
| 32 | 0.05 | 0.01 | 0.0637 | 0.3700 | 1.7142 |  |
| 64 | 0.02 | 0.001 | 0.0664 | 0.3780 | 1.9453 |  |
| 64 | 0.02 | 0.01 | 0.0695 | 0.3923 | 2.0414 | **selected** |
| 64 | 0.05 | 0.001 | 0.0604 | 0.3573 | 1.6369 |  |
| 64 | 0.05 | 0.01 | 0.0688 | 0.3918 | 1.8455 |  |

Selected: `{'factors': 64, 'lr': 0.02, 'reg': 0.01}`

Grids are coarse and the differences between the better settings are small; read the selections as reasonable defaults, not sharp optima.
Settings at the edge of a grid are flagged in the README.
