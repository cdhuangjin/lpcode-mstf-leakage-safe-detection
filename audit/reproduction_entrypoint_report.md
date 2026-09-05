# Phase 4 reviewer entry-point verification

Status: PASS on 5 September 2026. No frozen research output was changed.

| Command suffix for `python scripts/reproduce.py` | Output directory under audit/recomputed | CLI seconds | Status |
| --- | --- | ---: | --- |
| --mode smoke | smoke | 2.576 | PASS |
| --mode audit | audit | 56.343 | PASS |
| --mode table2 | table2 | 0.133 | PASS |
| --mode table3 | table3 | 0.129 | PASS |
| --mode all-saved | all-saved-001 | 0.179 | PASS |

All-saved was subsequently rerun into `all-saved-002` after adding explicit
mechanism-aggregate provenance and validation. That is the retained current
table/figure-data example. Earlier output directories remain locally intact;
the CLI always creates a new suffix rather than overwriting them.

These timings used the pinned Windows/Python 3.11 environment and already
validated feature/attack caches. They are not cold-start timing guarantees.
All-saved and table modes never fit a model. Audit reconstructs pair metadata
and checks validated caches, rather than reading a prior audit PASS flag.

## Actual smoke fit

The full C positive bank, seed 42, fold 0 of five was evaluated once with
fixed XGBoost A0 (20 dimensions) and A1 (30 dimensions). Scaler fitting uses
training data only through the original Pipeline. Measured positive-class F1:
A0 = 0.896551724137931, A1 = 0.9186206896551725, difference = 2.206896551724147 pp.
This single cell is not a replacement for the formal 60-cell clean mean.

The smoke directory contains actual configuration/model parameters,
environment, dataset hash, pair manifests without source snippets, metrics,
summary, and artifact hashes. `cache_reused` is true. The model-work section
took 1.389 seconds; total CLI elapsed was 2.576 seconds.

## Saved outputs and uncertainty scope

Tables 2–7 are available as CSV and Markdown. Eight figure-source CSVs cover
main results, variants, contrasts, transformations, generators, languages,
negative modes and mechanism aggregates. F1 means and paired differences
are recomputed from frozen per-cell ledgers, enforcing full matched coverage.
Gate D dispersion uses sample SD (ddof=1), consistent with the frozen table.
Intervals are retained from hash-verified summaries, not re-bootstrapped.
Mechanism values are retained frozen aggregates and cross-checked against
the separate grouped-importance CSV over all 64 environment/group cells;
no new importance fitting or per-fold importance reconstruction is claimed.

## Tests and independent review

The final full-suite command was:
`python -m pytest repro/2502.17749/v1/tests tests -q --junitxml=audit/hardened_tests.xml`.

481 passed in 109.32 seconds, with 19 unchanged dependency/MLP warnings:
423 original research tests, 4 initial release tests, 18 numerical-audit
tests, 22 isolation-audit tests and 14 reproduction-entry tests.
The JUnit XML preserves machine-readable outcomes.

Independent specification and code-quality reviews approved the entry points
after mechanism provenance and output-path error handling were corrected.
The original v1.0.1 file inventory is preserved separately as
`audit/v1.0.1_file_manifest.json`; later documentation edits must not be
confused with changes to frozen scientific evidence.
