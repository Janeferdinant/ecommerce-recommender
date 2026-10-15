# A/B test simulation 2026-10-14: UCI Online Retail II

**This is a simulation on historical data under stated assumptions. It is not a deployed experiment and its outputs are not
measured business impact.**

- Command: `python scripts/simulate_ab.py --dataset retail`; seed 42; 2000 simulated experiments per cell
- Control: popularity. Treatment: EASE (best validation NDCG@10); k = 10
- Population: the 2,191 held-out customers; each customer's outcome under each arm comes from an offline replay
- Outcome: the customer "converts" if at least one product in the arm's top 10 list is one they bought, new to them, in the
  test window. Replay conversion: control 28.8%, treatment 46.1% (relative +60.2%).
- Test: two-sided, alpha 0.05, pooled two-proportion z-test; power target 80%
- Assumptions: later purchases of listed products are attributed to the list; customers are independent; no novelty effect,
  seasonality or drift; the population only holds customers who bought something new, so these conversion rates are much
  higher than a live click-through or purchase rate. A live experiment needs its own baseline rate; the first table is
  therefore given for several baseline rates.

## Customers needed per arm (analytic) for a relative uplift in a conversion rate

| Baseline rate | +2% | +5% | +10% | +20% |
|---|---|---|---|---|
| 2.0% | 1,941,808 | 315,206 | 80,682 | 21,109 |
| 5.0% | 752,703 | 122,124 | 31,234 | 8,158 |
| 10.0% | 356,335 | 57,763 | 14,751 | 3,841 |
| 28.8% | 97,594 | 15,750 | 3,992 | 1,023 |

## What if only part of the replay effect is real?

Each customer in the treatment arm keeps the replay outcome with the given probability and otherwise behaves as in the control arm.

| Share of the replay effect assumed real | Implied relative uplift | Customers per arm needed | Simulated power at that size |
|---|---|---|---|
| 100.0% | +60.2% | 122 | 81.0% |
| 50.0% | +31.5% | 422 | 81.6% |
| 25.0% | +13.5% | 2,220 | 79.5% |

## Simulated power against sample size (baseline 28.8%)

| Customers per arm | Power if the replay effect is real | Power if half of it is real |
|---|---|---|
| 100 | 73.0% | 27.8% |
| 200 | 95.5% | 49.6% |
| 400 | 99.9% | 78.5% |
| 800 | 100.0% | 96.9% |
| 1,600 | 100.0% | 100.0% |
| 3,200 | 100.0% | 100.0% |

## Smallest detectable relative uplift at baseline 28.8%

| Customers per arm | Minimum detectable uplift (80% power) |
|---|---|
| 500 | +28.9% |
| 1,000 | +20.2% |
| 2,000 | +14.2% |
| 5,000 | +8.9% |
| 10,000 | +6.3% |

A/A check (both arms see popularity): false-positive rate 4.5% with 200 per arm and 5.4% with 1,000 per arm; it should be close to 5%.

## Revenue per customer (continuous outcome, same replay)

Mean replay Revenue@10: control 1.487, treatment 2.069 GBP per customer
(sd about 3.49; the distribution is heavy-tailed, so a Welch test with a normal reference is only a
rough guide). Detecting the replay difference of 0.582 GBP needs about 565 customers per arm
(analytic); simulated power at that size: 80.4%. A real test would be run on
actual order value, which is noisier than this proxy.

| Customers per arm | Simulated power (replay revenue effect) |
|---|---|
| 100 | 23.0% |
| 200 | 40.9% |
| 400 | 67.0% |
| 800 | 90.7% |
| 1,600 | 99.8% |
| 3,200 | 100.0% |
