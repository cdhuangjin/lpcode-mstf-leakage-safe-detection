# Phase 1 independent-environment test report

Status: PASS. Date: 5 September 2026.

Fresh public clone of v1.0.1 (`76209b7345a4ec6bd3c1ea3f9a24c5aa65583029`),
new `.venv`; packages installed from root `requirements.txt` and editable
`repro/2502.17749/v1`. Pinned upstream fetched using the public script.
No author-side caches or models were copied into this checkout.

| Item | Observed |
| --- | --- |
| Python | 3.11.15 |
| pip | 24.0 |
| OS | Windows-10-10.0.26200-SP0 (platform API string) |
| Dependency health | `pip check`: No broken requirements found |
| Total collected / passed | 427 / 427 |
| Research suite / release suite | 423 / 4 |
| Failed / errors / skipped / xfailed | 0 / 0 / 0 / 0 |
| Warnings | 19 |
| pytest elapsed | 108.17 seconds |
| JUnit suite elapsed | 107.705 seconds |

All 26 release pins match installed package versions, including NumPy 1.23.5,
scikit-learn 1.2.0, XGBoost 2.1.4, tree-sitter 0.26.0, C grammar 0.24.2,
C++ grammar 0.23.4, Java grammar 0.23.5 and pytest 8.4.2.
The complete pin list remains `requirements.txt`.

Command: `python -m pytest repro/2502.17749/v1/tests tests -q --junitxml=audit/baseline_tests.xml`.
Machine-readable evidence: baseline_tests.xml; full output: baseline_tests.log.

Warnings are Matplotlib/pyparsing deprecations and small-fixture MLP
convergence warnings at the unchanged 200-iteration limit. They were not
suppressed or treated as newly converged models. Two tests in test_t3.py
exercise Windows-specific lock retry paths and would skip on other OSes.
No Linux/macOS test execution or full formal training replication is claimed.
