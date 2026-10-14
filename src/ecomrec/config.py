"""Default model settings.

``ALS_PARAMS`` are the values selected by ``scripts/tune_als.py`` on a
validation window carved out of the training period (the test period is never
used for tuning). See the dated tuning log in ``docs/runs/``.
"""

ALS_PARAMS: dict[str, float | int] = {"factors": 32, "iterations": 15, "reg": 0.1, "alpha": 10.0}
