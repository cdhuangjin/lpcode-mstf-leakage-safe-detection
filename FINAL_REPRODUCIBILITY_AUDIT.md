# Final reproducibility audit and handoff
Date: 2026-09-05. **Reproducibility gate PASS; external validation BLOCKED.**

Publication recheck: the complete suite again passed 481 tests (19 warnings, 83.99 s; zero failures/errors/skips), recorded in audit/final_verification_tests.xml. The refreshed combined audit passed in 49.757 s with existing caches at audit/recomputed/publication-final/report.json. Earlier timings below identify their original executions, not this repeat.

## What was actually verified
A fresh public v1.0.1 clone (76209b7345a4ec6bd3c1ea3f9a24c5aa65583029) and new Python 3.11.15 environment were used. The original research workspace was preserved. The 26 pinned dependencies were checked; pip check passed.

| Check | Outcome | Evidence |
| --- | --- | --- |
| Original public suite | 427 passed: 423 research + 4 release tests | audit/baseline_tests.xml; audit/test_report.md |
| Hardened suite | 481 passed, 19 warnings, no failures/errors/skips; 109.32 s | audit/hardened_tests.xml |
| Saved headline and ablation arithmetic | 58 checks PASS | audit/manuscript_evidence_audit.json |
| Raw pair reconstruction | 2,928 A–D ledger records matched | audit/isolation_audit.json |
| Origin and exact-code overlap | Zero combined train/test endpoint overlap under the audited identity rules | audit/isolation_audit.md |
| A0–A5 pairing | 1,800 rows bind to formal A/B manifests | audit/isolation_audit.json |
| Actual smoke fitting | C, seed 42, fold 0/5, fixed A0/A1 | audit/recomputed/smoke/ |
| Table regeneration | Tables 2–7 plus eight figure-source CSVs | audit/recomputed/all-saved-002/ |
| Reviewer commands | smoke, audit, table2, table3, all-saved | scripts/reproduce.py; reproducibility.md |

The smoke F1 scores were 0.8965517 and 0.9186207. These are single-fold pipeline checks, not replacements for paper means.

## Frozen headline values
| Gate | Comparator | Proposed arm | F1 comparator | F1 proposed | Difference (pp) |
| --- | --- | --- | ---: | ---: | ---: |
| A | A0, original endpoints, matched XGBoost | A1, original endpoints + signed difference | 0.9149 | 0.9317 | +1.683 |
| B | LPcode original MLP | A5 | 0.8915 | 0.9671 | +7.563 |
| C combined transformation | LPcode original MLP | A5 | 0.8415 | 0.9531 | +11.160 |
| D | LPcode original MLP | A5 | 0.8938 | 0.9652 | +7.141 |

Gate A's headline is not full A5. Metrics are means of positive-class F1 over the specified evaluation cells, not class-macro F1. B–D comparisons are whole-system contrasts, not isolated relative-block effects.

## Boundaries and integrity
The audit reconstructs raw splits/features and independently aggregates saved cell metrics; it does not refit all historical models. Frozen confidence intervals are retained, not newly bootstrapped. Mechanism aggregates are hash-bound and cross-checked, not new permutation runs. Gate D origins are language-scoped; exact code comparisons are global. Three-seed uncertainty and semantic-duplicate limitations remain.

Two recovered historical t3 source snapshots match the registered full-file hashes; the live t3 file is not falsely claimed identical. See audit/historical_sources/README.md. The original inventory is retained in audit/v1.0.1_file_manifest.json. Original implementation, evidence and manuscript bytes were preserved; README and ignore rules were deliberately updated. No upstream raw code/cache is included in the release.

Author-new software uses the authorized MIT grant within LICENSE scope. Data, manuscript, figures and inherited materials are not newly relicensed. No archival DOI or new tagged version is claimed.

## Reviewer quick start
Use Python 3.11, the pinned environment and upstream acquisition steps in [reproducibility.md](reproducibility.md). After installation, from the repository root:

```powershell
python scripts/fetch_upstream.py
python scripts/reproduce.py --mode audit
python scripts/reproduce.py --mode smoke
python scripts/reproduce.py --mode all-saved
python -m pytest tests repro/2502.17749/v1/tests -q
```

Fresh output directories are under audit/recomputed; per-run artifact_hashes.json binds outputs. Outputs are not silently overwritten. Cached CLI observations: smoke 2.576 s, audit 56.343 s, Table 2 0.133 s, Table 3 0.129 s. These are not uncached installation or full-training timings.

## Remaining work
See [CodeMirage status](CODEMIRAGE_EXTERNAL_VALIDATION.md). Internal paper plans have been removed from the current public branch. A full historical refit, external Gate E, its adapter/training/table/failure-case scripts and external artifact CSV are not represented as completed. The existing smoke and regenerated outputs already carry SHA-256 manifests; a separate external hash-export utility remains deferred with external experiment execution.
