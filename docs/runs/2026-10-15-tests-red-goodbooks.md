# Tests-first red run: second dataset loader (2026-10-15)

`tests/test_goodbooks.py` was written before the Goodbooks-10k loader and the model-exclusion helper existed, so collection fails.
The other modules of the study were carried over together with their tests (137 passing before this step).

- Command: `python -m pytest -q tests/test_goodbooks.py`

```
E   ImportError: cannot import name 'goodbooks_interactions' from 'ecomrec.data' (<path>/Downloads\portfolio-audit\work3\recommender-h\src\ecomrec\data.py)
ERROR tests/test_goodbooks.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.42s
```
