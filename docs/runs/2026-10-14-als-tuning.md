# ALS tuning run 2026-10-14

Hyperparameters were chosen on a validation window inside the training period.
The test period was not used (and the first test timestamp, 2011-09-12 12:27:00, is after the validation window ends).

- Started: 2026-10-14 04:30 UTC
- Command: `python scripts/tune_als.py --seed 42`
- Seed: 42; alpha held at 10.0
- Fit window: 490,156 records, 2009-12-01 07:45:00 to 2011-04-21 16:26:00
- Validation window: 122,661 records, 2011-04-21 16:37:00 to 2011-09-12 12:24:00
- Grid: factors [16, 32, 64], regularisation [0.01, 0.1, 1.0], iterations [10, 15] (18 fits)
- Selection rule: highest validation NDCG@10 (earliest grid point wins ties)
- Wall time: 168 s on CPU

| factors | reg | iterations | Precision@10 | Recall@10 | NDCG@10 | |
|---|---|---|---|---|---|---|
| 16 | 0.01 | 10 | 0.0628 | 0.0369 | 0.0728 |  |
| 16 | 0.01 | 15 | 0.0629 | 0.0366 | 0.0725 |  |
| 16 | 0.1 | 10 | 0.0629 | 0.0369 | 0.0728 |  |
| 16 | 0.1 | 15 | 0.0629 | 0.0366 | 0.0726 |  |
| 16 | 1.0 | 10 | 0.0631 | 0.0370 | 0.0730 |  |
| 16 | 1.0 | 15 | 0.0631 | 0.0366 | 0.0726 |  |
| 32 | 0.01 | 10 | 0.0651 | 0.0385 | 0.0750 |  |
| 32 | 0.01 | 15 | 0.0657 | 0.0395 | 0.0751 |  |
| 32 | 0.1 | 10 | 0.0651 | 0.0385 | 0.0750 |  |
| 32 | 0.1 | 15 | 0.0656 | 0.0394 | 0.0750 |  |
| 32 | 1.0 | 10 | 0.0652 | 0.0385 | 0.0751 |  |
| 32 | 1.0 | 15 | 0.0657 | 0.0395 | 0.0752 |  |
| 64 | 0.01 | 10 | 0.0651 | 0.0406 | 0.0744 |  |
| 64 | 0.01 | 15 | 0.0657 | 0.0415 | 0.0755 |  |
| 64 | 0.1 | 10 | 0.0651 | 0.0406 | 0.0744 |  |
| 64 | 0.1 | 15 | 0.0656 | 0.0415 | 0.0754 |  |
| 64 | 1.0 | 10 | 0.0652 | 0.0406 | 0.0746 |  |
| 64 | 1.0 | 15 | 0.0656 | 0.0416 | 0.0756 | **selected** |

Selected: {'factors': 64, 'reg': 1.0, 'iterations': 15}

The grid is coarse and validation differences between the better settings are small; treat the choice as a
reasonable default, not a sharp optimum.
