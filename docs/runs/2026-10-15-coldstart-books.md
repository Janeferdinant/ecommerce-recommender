# Cold-start run 2026-10-15: Goodbooks-10k (subset)

Status: Verified (run 2026-10-15). Offline simulation of new customers; not measured business impact.

- Started: 2026-10-15 17:23 UTC
- Command: `python scripts/run_coldstart.py --dataset books`
- Protocol: temporal split on file order (the ratings carry no timestamps): the earliest 80% of the subset's ratings train, the latest 20% test. Cold customers: 1,023 (a random 25% of the evaluation customers, seed 42).
- Method: for n known items the customers' other training interactions are removed and every model is refitted;
  recommendations come from the n known items only (the models' basket mode). The first n items are the earliest
  by time, ties by item id. Test items are unchanged. One seed; wall time 387 s on CPU.
- Values: mean over the cold customers with 95% bootstrap intervals over customers (1,000 resamples), k = 10.

### Hit rate@10 (%)

| Model | 1 known item | 2 known items | 3 known items | full history |
|---|---|---|---|---|
| Popularity | 17.1 (14.9 to 19.6) | 17.1 (14.9 to 19.6) | 17.2 (15.0 to 19.6) | 19.3 (16.9 to 21.7) |
| Item-based CF | 15.2 (13.1 to 17.4) | 16.3 (14.1 to 18.6) | 16.4 (14.3 to 18.9) | 25.7 (23.2 to 28.3) |
| Co-purchase (basket co-occurrence) | 15.3 (13.2 to 17.5) | 16.1 (14.1 to 18.4) | 15.0 (12.8 to 17.3) | 29.2 (26.5 to 32.1) |
| Implicit ALS | 14.7 (12.5 to 16.8) | 14.8 (12.8 to 16.8) | 15.6 (13.5 to 17.8) | 32.8 (29.9 to 36.0) |
| EASE | 15.0 (12.8 to 17.1) | 15.5 (13.5 to 17.8) | 15.8 (13.7 to 18.0) | 36.2 (33.4 to 39.3) |
| BPR matrix factorisation | 16.6 (14.1 to 19.2) | 17.1 (14.9 to 19.6) | 17.7 (15.2 to 20.1) | 31.8 (28.8 to 34.6) |

### NDCG@10

| Model | 1 known item | 2 known items | 3 known items | full history |
|---|---|---|---|---|
| Popularity | 0.0352 (0.0292 to 0.0418) | 0.0352 (0.0293 to 0.0418) | 0.0355 (0.0296 to 0.0424) | 0.0471 (0.0396 to 0.0554) |
| Item-based CF | 0.0275 (0.0224 to 0.0328) | 0.0299 (0.0245 to 0.0354) | 0.0306 (0.0250 to 0.0361) | 0.0611 (0.0531 to 0.0701) |
| Co-purchase (basket co-occurrence) | 0.0281 (0.0229 to 0.0333) | 0.0290 (0.0239 to 0.0340) | 0.0283 (0.0228 to 0.0337) | 0.0675 (0.0589 to 0.0774) |
| Implicit ALS | 0.0272 (0.0222 to 0.0326) | 0.0273 (0.0222 to 0.0325) | 0.0287 (0.0233 to 0.0342) | 0.0711 (0.0627 to 0.0801) |
| EASE | 0.0272 (0.0223 to 0.0328) | 0.0285 (0.0236 to 0.0341) | 0.0289 (0.0237 to 0.0345) | 0.0731 (0.0652 to 0.0820) |
| BPR matrix factorisation | 0.0320 (0.0263 to 0.0383) | 0.0345 (0.0284 to 0.0409) | 0.0327 (0.0270 to 0.0386) | 0.0678 (0.0592 to 0.0767) |

Read each column against the popularity row, which ignores the known items, and against the full-history column. Popularity
never uses the known items, so its small differences between columns come only from which items are excluded as already known.
