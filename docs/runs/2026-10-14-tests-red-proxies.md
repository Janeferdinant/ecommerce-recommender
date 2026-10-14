# Test-first run (red): engagement/revenue proxies, tuning and bootstrap

- Run: 2026-10-14 04:24 UTC
- Command: ```python -m pytest -q```
- Tests for `hit_rate_at_k`, `revenue_at_k`, `catalogue_coverage`, `relative_uplift`, `evaluate_detailed`, `ecomrec.stats`, `ecomrec.tuning` and the `price` column were written before the implementation.

```text

=================================== ERRORS ====================================
___________________ ERROR collecting tests/test_metrics.py ____________________
ImportError while importing test module '<repo>\tests\test_metrics.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_metrics.py:5: in <module>
    from ecomrec.metrics import (
E   ImportError: cannot import name 'catalogue_coverage' from 'ecomrec.metrics' (<repo>\src\ecomrec\metrics.py)
___________________ ERROR collecting tests/test_proxies.py ____________________
ImportError while importing test module '<repo>\tests\test_proxies.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_proxies.py:6: in <module>
    from ecomrec.evaluate import evaluate, evaluate_detailed
E   ImportError: cannot import name 'evaluate_detailed' from 'ecomrec.evaluate' (<repo>\src\ecomrec\evaluate.py)
____________________ ERROR collecting tests/test_tuning.py ____________________
ImportError while importing test module '<repo>\tests\test_tuning.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\Lib\importlib\__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_tuning.py:1: in <module>
    from ecomrec.tuning import tune_als, validation_split
E   ModuleNotFoundError: No module named 'ecomrec.tuning'
=========================== short test summary info ===========================
ERROR tests/test_metrics.py
ERROR tests/test_proxies.py
ERROR tests/test_tuning.py
!!!!!!!!!!!!!!!!!!! Interrupted: 3 errors during collection !!!!!!!!!!!!!!!!!!!
3 errors in 0.65s
```

