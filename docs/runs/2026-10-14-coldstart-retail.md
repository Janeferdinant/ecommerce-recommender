# Cold-start run 2026-10-14: UCI Online Retail II

Status: Verified (run 2026-10-14). Offline simulation of new customers; not measured business impact.

- Started: 2026-10-14 21:53 UTC
- Command: `python scripts/run_coldstart.py --dataset retail`
- Protocol: temporal split: the earliest 80% of purchase records train, the latest 20% test. Cold customers: 548 (a random 25% of the evaluation customers, seed 42).
- Method: for n known items the customers' other training interactions are removed and every model is refitted;
  recommendations come from the n known items only (the models' basket mode). The first n items are the earliest
  by time, ties by item id. Test items are unchanged. One seed; wall time 324 s on CPU.
- Values: mean over the cold customers with 95% bootstrap intervals over customers (1,000 resamples), k = 10.

### Hit rate@10 (%)

| Model | 1 known item | 2 known items | 3 known items | full history |
|---|---|---|---|---|
| Popularity | 20.6 (17.3 to 24.1) | 20.6 (17.3 to 24.1) | 20.8 (17.7 to 24.3) | 26.5 (22.6 to 30.1) |
| Item-based CF | 14.1 (11.1 to 17.0) | 16.2 (13.3 to 19.5) | 18.4 (15.1 to 21.7) | 34.5 (30.5 to 38.5) |
| Item-based CF, recency-weighted | 9.1 (6.7 to 11.5) | 9.3 (6.9 to 11.7) | 12.2 (9.5 to 15.0) | 37.6 (33.8 to 41.6) |
| Co-purchase (basket co-occurrence) | 14.4 (11.7 to 17.3) | 16.2 (13.1 to 19.5) | 16.2 (13.1 to 19.5) | 34.7 (30.8 to 38.7) |
| Co-purchase + price-aware re-ranking | 10.9 (8.2 to 13.7) | 11.1 (8.4 to 13.9) | 12.2 (9.7 to 15.1) | 14.4 (11.7 to 17.5) |
| Implicit ALS | 15.7 (12.8 to 18.8) | 16.1 (13.0 to 19.0) | 17.9 (14.8 to 21.0) | 37.2 (33.2 to 41.4) |
| EASE | 15.1 (12.0 to 18.1) | 16.2 (13.1 to 19.2) | 20.3 (16.8 to 23.5) | 45.1 (40.9 to 49.3) |
| BPR matrix factorisation | 20.3 (16.8 to 23.7) | 22.8 (19.2 to 26.3) | 20.4 (17.2 to 24.1) | 34.3 (30.5 to 38.3) |

### NDCG@10

| Model | 1 known item | 2 known items | 3 known items | full history |
|---|---|---|---|---|
| Popularity | 0.0300 (0.0242 to 0.0359) | 0.0300 (0.0243 to 0.0361) | 0.0303 (0.0245 to 0.0363) | 0.0413 (0.0343 to 0.0488) |
| Item-based CF | 0.0207 (0.0157 to 0.0262) | 0.0259 (0.0199 to 0.0326) | 0.0271 (0.0215 to 0.0334) | 0.0623 (0.0531 to 0.0713) |
| Item-based CF, recency-weighted | 0.0136 (0.0093 to 0.0180) | 0.0131 (0.0093 to 0.0174) | 0.0185 (0.0133 to 0.0237) | 0.0764 (0.0657 to 0.0867) |
| Co-purchase (basket co-occurrence) | 0.0227 (0.0175 to 0.0288) | 0.0232 (0.0178 to 0.0286) | 0.0238 (0.0179 to 0.0300) | 0.0637 (0.0545 to 0.0733) |
| Co-purchase + price-aware re-ranking | 0.0150 (0.0102 to 0.0207) | 0.0159 (0.0112 to 0.0219) | 0.0170 (0.0120 to 0.0230) | 0.0233 (0.0178 to 0.0301) |
| Implicit ALS | 0.0235 (0.0182 to 0.0295) | 0.0261 (0.0203 to 0.0320) | 0.0288 (0.0228 to 0.0351) | 0.0662 (0.0572 to 0.0761) |
| EASE | 0.0223 (0.0168 to 0.0276) | 0.0265 (0.0202 to 0.0335) | 0.0294 (0.0233 to 0.0362) | 0.0863 (0.0755 to 0.0971) |
| BPR matrix factorisation | 0.0295 (0.0239 to 0.0355) | 0.0322 (0.0262 to 0.0390) | 0.0304 (0.0246 to 0.0372) | 0.0552 (0.0473 to 0.0633) |

Read each column against the popularity row, which ignores the known items, and against the full-history column. Popularity
never uses the known items, so its small differences between columns come only from which items are excluded as already known.
