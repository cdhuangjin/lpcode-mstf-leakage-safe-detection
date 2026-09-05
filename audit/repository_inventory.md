# Phase 0 repository inventory

Audit baseline: public v1.0.1, commit `76209b7345a4ec6bd3c1ea3f9a24c5aa65583029`.
Fresh GitHub clone: `release/submission-audit-20260905` under the author workspace.
Work is isolated from the author experiment tree and existing release clone.

## Structure and entry points

| Area | Actual location / entry |
| --- | --- |
| Implementation | `repro/2502.17749/v1/lpcode_v1/` (31 Python modules) |
| Original endpoint extractor | `features_official.analyze_code` (10 features; inherited rights excluded from MIT) |
| Enhanced endpoint extractor | `features_enhanced.py`; `t3.load_or_build_enhanced_cache` (28 total) |
| Pair construction | `t3._build_pair_splits`, `build_t1_pair_splits`, `build_t3_splits`; `t5.build_language_pair_bank` |
| Representation | `representations.build_representation`; relative denominator `abs(human)+1e-8` |
| Models / evaluation | `experiment.build_model`, `evaluate_fold`; train-only Pipeline scaler |
| Formal experiments | `t1_strict`, `t3`, `t4`, `t5` module CLIs |
| A0–A5 | `ablation.py` and `gates_ablation.METHOD_CONTRACT` |
| Evidence | Gate `folds.jsonl`, configs, manifests; `canonical.py`, Gate validators, `paper_audit.py`, `integrity_audit.py` |
| Negative sensitivity | `negative_pair_robustness.py`; `results/negative_pair_robustness/raw_results.json` (not folds.jsonl) |
| Tables / figures | `paper_assets.py`, `tools/revision_evidence.py`, `tools/revision_figures.py` |
| Tests | 29 research test modules under `repro/.../tests`, plus `tests/test_release.py` |
| Data acquisition | `scripts/fetch_upstream.py`; pinned upstream, ignored `repro/2502.17749/code` |
| Environment | root `requirements.txt` is authoritative release pin list; nested `pyproject.toml`; historical `uv.lock` is not the release install route |
| Documentation | README, `docs/{REPRODUCIBILITY,DATA_ACCESS,STATISTICS,RELEASE_STATUS}.md` |

No top-level src/configs/manifests/evidence/reproduction/supplementary folders
are required: equivalent material is already nested under repro/results.
No requirements-lock.txt or environment.yml is present; do not invent one as
a prerequisite. Manuscript files are under `results/07_manuscript`.

## Initial findings to verify

1. Existing no-fit verifier hashes frozen Gate files and checks ledger counts,
   but reads stored strict PASS fields. This alone is not independent
   raw-code reconstruction of dual-endpoint or exact-content isolation.
2. Frozen configs contain historical Windows absolute paths. Public readers
   use `release_paths.resolve_recorded_path`; training and manuscript tools
   do not universally use it. Fresh-clone execution must determine impact.
3. `tools/revision_evidence.py` expects author-only original manuscripts and
   uses literal registry paths. It is not a reviewer one-click table builder.
4. Existing `smoke.py` validates extraction/grouping without training.
   `t1_strict.run_smoke` trains 16 cells, not the requested one-fold A0/A1
   smoke. Reuse its components without changing the frozen runner.
5. Feature caches and per-pair predictions are not in the public deposit.
   Raw acquisition / deterministic reconstruction is required before treating
   cache and endpoint audits as independently checked.
6. Historical tests report 423; current collection, outcomes, warnings and
   skipped/xfail counts are to be measured afresh, not copied.
7. MIT is scoped to new author software; raw upstream data must not be
   redistributed. Data remains without an additional reuse licence.

This is an entry-point and contract inventory, not a claim that all code
paths or every historical author-side file have been exercised.
