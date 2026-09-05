# Reproducibility guide

This guide distinguishes saved-evidence regeneration, raw-data isolation auditing, an actual smoke fit and full formal retraining. Follow the Python 3.11 installation in [README.md](README.md); `requirements.txt` is canonical, while the older `uv.lock` is historical provenance. Run commands from the repository root after activating that environment.

## Data acquisition and provenance

The supported acquisition helper is:

```bash
python scripts/fetch_upstream.py
```

It fetches `https://github.com/Shinwoo-Park/LPcode.git` at `b3660c8262ae57e14498528119607ee673d4257a` into `repro/2502.17749/code`. Its equivalent raw checkout operation, when that directory does not already exist, is:

```bash
git clone --filter=blob:none --no-checkout https://github.com/Shinwoo-Park/LPcode.git repro/2502.17749/code
git -C repro/2502.17749/code checkout --detach b3660c8262ae57e14498528119607ee673d4257a
```

Use the helper for the supported workflow: it additionally stages deposited baseline metric files only if absent or identical. An existing checkout at another revision is rejected. Task 1 data are `repro/2502.17749/code/experiment/task1/dataset/{c,cpp,java,py}.jsonl`; record counts are 3,656 / 3,080 / 11,952 / 15,480 respectively. Exact SHA-256 values and byte sizes are in [UPSTREAM_CHECKSUMS.json](docs/UPSTREAM_CHECKSUMS.json). See [DATA_ACCESS.md](docs/DATA_ACCESS.md) for the numerical dictionary and Task 2 context.

No open upstream licence was identified at the recorded release check. Acquisition directly from the owner does not grant reuse rights; raw snippets and original upstream program bodies are excluded from this deposit. The inherited compatibility implementation `features_official.py` retains its explicit licence exclusion.

## Pipeline and pair construction

The logical dependencies are: raw file/row/code hashing → source-origin components → component partitioning → partition-local negative construction → endpoint features → pair representations → training-only scaling and model fit → per-cell ledger → paired aggregation → tables and figure data.

The implementation may compute and cache deterministic per-snippet endpoint features before constructing pairs. That cache construction does not fit a dataset-level transform; learned scaling is fitted only on the training matrix. Distinguish this physical execution order from the logical data-isolation dependencies above.

Exact endpoint-code matches connect source origins into components. Component assignment uses deterministic hash tie-breaking; train and test components are disjoint. Positives come from the positive bank. Default negatives are constructed separately within each partition by cross-component cyclic derangement, preserving the partition boundary. Current/random/hard negative sensitivity is a separate original10 A1-versus-A0 analysis, not enhanced112 full MSTF. Gate C additionally checks transformed candidate code against training endpoints. These are exact-origin/code checks, not proof that all semantic similarity has been removed.

The raw-data audit binds 2,928 frozen Gate records (A=480, B=960, C=1,440, D=48) and 1,800 A0–A5 records. Recorded reconstruction has zero origin and exact-code overlap, including transformed-code checks on all 1,440 Gate C records. Gate D origin identifiers are language-scoped; exact-code hashes are compared globally across languages. The origin check therefore does not establish global semantic program identity. A passing digest comparison establishes that reconstructed default pairs match frozen contracts; it does not establish replication of fitted predictions or model scores.

Current `t3.py` contains later negative-pair extensions. The [historical-source archive](audit/historical_sources/README.md) records two recovered source files with exact SHA-256 matches to Gate A and Gate B/C/D bindings. They are archival evidence, not replacement runtime modules. Do not copy them over the live package or resume frozen ledgers. Historical retraining requires an isolated, version-matched package and new output tree.

## Supported reproduction commands

| Command | Raw upstream needed? | Model fitting? | Scope |
| --- | --- | --- | --- |
| `python scripts/quickstart.py` | No | No | Synthetic representation example |
| `python scripts/verify_release.py` | No | No | Frozen registry/binding/arithmetic checks |
| `python scripts/reproduce.py --mode table2` | No | No | Table 2 CSV/Markdown |
| `python scripts/reproduce.py --mode table3` | No | No | Table 3 CSV/Markdown |
| `python scripts/reproduce.py --mode all-saved` | No | No | Tables 2–7 and eight figure CSVs |
| `python scripts/reproduce.py --mode audit` | Yes | No | Manuscript evidence plus raw pair/isolation reconstruction |
| `python scripts/reproduce.py --mode smoke` | Yes | Yes | C, seed 42, fold 0/5, A0/A1 |

`audit` calls `audit_manuscript_evidence.run_audit` and `audit_isolation.run_audit` (the Phase 2 and Phase 3 checks). It validates or builds raw-derived caches under `audit/cache/`. `all-saved` validates ledger bytes against `FILE_MANIFEST.json`, recomputes numerical cell aggregates and checks applicable frozen summary values. It retains frozen bootstrap bounds rather than drawing new bootstrap samples. Mechanism values are retained aggregates whose schema, coverage, registry binding and values are cross-checked between `mechanism_summary.json` and `grouped_importance.csv`; this does not recompute importance from per-fold data.

The smoke uses the same first-ten-feature pair matrices and fixed XGBoost implementation as the strict experiment: 300 estimators, maximum depth 4, learning rate .05, subsample .8, column subsample .8, `n_jobs=1`, and train-only `StandardScaler`. It records full parameter/environment contracts and train/test pair manifests. Observed A0/A1 F1 was .8965517/.9186207 for this one fold, not the paper-level mean.

## Full-training boundary and runner map

Full Gate A–D / A0–A5 retraining was not rerun during this audit. The commands below expose the existing research workflow; they are not an automated clean-room historical replay. Runners bind preceding gates and implementation hashes, and several dependencies use canonical result locations. A generic `--output-root` override alone does not isolate every dependency.

Prepare a **separate rerun checkout** with a separately installed package and acquired pinned upstream. Preserve the deposited checkout unchanged; never empty its result directories. In the separate rerun checkout, first preserve its copies of `results/` and `repro/2502.17749/v1/results/` in an explicitly named, recoverable backup location outside the new canonical output tree. After verifying that backup, the separate rerun checkout's canonical result directories must start absent or empty; copying a backup while retaining historical ledgers in those directories is insufficient. Inspect archived configurations and the required source version before fitting. Prepare new baseline evidence and prerequisite manifests at canonical paths; do not mix old ledgers with new fits. Historical version matching is a manual prerequisite, not something the public reproduction CLI performs.

The original baseline requires upstream `experiment/task1/main.py` and `experiment/task2/main.py`, executed from each respective directory, for each `--lang c`, `cpp`, `java`, `py`, with `--seed 42 --k 5`. These generate the eight baseline output files expected by `lpcode_v1.manifest`. The helper's copied frozen metrics are evidence for verification, not a newly fitted baseline. After those original runs, return to the repository root and build a fresh baseline manifest with `python -m lpcode_v1.manifest`. Only load trusted, locally generated pickle outputs.

After baseline prerequisites in the isolated checkout, the runner sequence is:

```bash
python -m lpcode_v1.t1_strict
python -m lpcode_v1.gates --output-root results/01_transition_test_strict_origins
python -m lpcode_v1.t3
python -m lpcode_v1.t4
python -m lpcode_v1.t5
python -m lpcode_v1.canonical --cross-language-root repro/2502.17749/v1/results/04_cross_language --write-state
python -m lpcode_v1.negative_pair_robustness --output-root results/negative_pair_robustness
python -m lpcode_v1.ablation run --registry results/06_paper_assets/frozen_result_registry.json --output-root results/05_mechanism_analysis
python -m lpcode_v1.mechanism attack --registry results/06_paper_assets/frozen_result_registry.json --t4-root results/03_style_attack
python -m lpcode_v1.mechanism importance --registry results/06_paper_assets/frozen_result_registry.json
```

| Stage | Module | Dependency / output role |
| --- | --- | --- |
| Gate A | `t1_strict`, then `gates` | Strict-origin original10 experiment and PASS evidence |
| Gate B | `t3` | Held-out generator; requires strict Gate A evidence |
| Gate C | `t4` | Deterministic transformations; requires preceding gate evidence |
| Gate D | `t5` | Held-out language; requires preceding gate evidence; not five-fold |
| Registry | `canonical` | New frozen-result registry for this rerun |
| Negative sensitivity | `negative_pair_robustness` | Original10 endpoints versus signed transition |
| A0–A5 | `ablation` | Complete Gate A/B caches and saved split contracts |
| Mechanism | `mechanism` | Attack decomposition and importance; importance may fit models |

Run each module with `--help` for actual cache/output options and review [the older command map](docs/REPRODUCIBILITY.md) alongside this boundary. The original non-strict `t1` is not a substitute for Gate A. Newly fitted results need their own audits; a source hash match or saved-result PASS is insufficient.

## Table and figure source map

Paths below are relative to the repository root. Ledgers are JSONL except negative sensitivity (`raw_results.json`); summaries are JSON unless marked CSV.

| Manuscript item / generated source | Row evidence | Retained summary / contract |
| --- | --- | --- |
| Table 1, Figures 1–2 | Feature/representation source and Gate configuration | `repro/2502.17749/v1/lpcode_v1/`; frozen Gate `config.json` files |
| Table 2, `figure_main.csv`, Figure 3 | `results/01_transition_test_strict_origins/folds.jsonl`; `results/02_unseen_llm/folds.jsonl`; `results/03_style_attack/folds.jsonl`; `repro/2502.17749/v1/results/04_cross_language/folds.jsonl` | `results/05_mechanism_analysis/ablation_summary.json`; Gate B/C/D `summary.json` |
| Table 3, `figure_variants.csv`, `figure_contrasts.csv`, Figure 4 left | `results/05_mechanism_analysis/folds.jsonl` | `results/05_mechanism_analysis/ablation_summary.json` |
| Table 4, `figure_generator.csv`, Figure 5 right | `results/02_unseen_llm/folds.jsonl` | `results/02_unseen_llm/summary.json` |
| Table 5, `figure_transformations.csv`, Figure 5 left | `results/03_style_attack/folds.jsonl` | `results/03_style_attack/summary.json` |
| Table 6, `figure_language.csv`, Figure 5 right | `repro/2502.17749/v1/results/04_cross_language/folds.jsonl` | Same directory `summary.json` |
| Table 7, `figure_negative.csv` | `results/negative_pair_robustness/raw_results.json` | `results/negative_pair_robustness/summary.csv` |
| `figure_mechanism.csv`, Figure 4 right | No per-fold importance recomputation | `results/05_mechanism_analysis/feature_importance/mechanism_summary.json` and `grouped_importance.csv` |

`python tools/revision_figures.py` renders the five revised figures from existing publication evidence, including `results/07_manuscript/revised_sources/evidence.json`; the reproduction CLI emits source CSVs only. The rendering tool can update figure/QA files in `results/07_manuscript/`, so it does not share the CLI's separate-output guarantee. Document typesetting may need Pandoc, OfficeCLI and an author-provided/licensed Wiley template; it is not required for numerical reproduction.

## Outputs, formats and hash scope

Every reproduction run reserves `audit/recomputed/<mode>/`, or a requested child directory supplied with `--output`. Existing paths receive a numeric suffix (`-001`, `-002`, …). Run artifacts never target the frozen result tree. Audit/smoke caches are separately placed in `audit/cache/`.

All modes write `report.json` (status and elapsed time) and `artifact_hashes.json`. Saved modes write `provenance.json`, CSV numerical tables, Markdown manuscript tables and, for `all-saved`, eight figure CSVs. Smoke also writes `config.json`, `environment.json`, `dataset_manifest.json`, `pair_manifest.json` and `.csv`, `metrics.json` and `.csv`, `summary.json` and `manifest.json`.

`artifact_hashes.json` is SHA-256 over every other file in that run directory. Smoke's inner `manifest.json` covers the smoke artifacts present before the CLI adds its final report/outer manifest; its own contents are excluded from its inner list. `provenance.json` records source hashes and explicitly distinguishes recomputed metrics from retained intervals/importance. `FILE_MANIFEST.json` authenticates listed release input bytes; `SOURCE_INVENTORY.json` records original-to-release source provenance. These scopes differ. Elapsed time and machine/environment records can change, so output-manifest byte equality across machines is not promised.

Frozen JSON/JSONL can contain historical machine paths. The documented portable path reader maps the recorded prefix at read time without rewriting source bytes. [DATA_ACCESS.md](docs/DATA_ACCESS.md) explains metric names, units and missing values.

## Determinism and interpretation

Formal seeds are 42, 123, 2024. Five grouped folds apply where specified; Gate D instead holds out a language. Deterministic component-hash tie-breaking and partition-local cyclic derangement bind pairs. Fixed XGBoost settings are listed above and in frozen configurations; full A5 uses relative epsilon `1e-8`. Library/platform differences and source changes must be recorded when evaluating newly fitted output.

F1 is positive-class F1, averaged equally over specified cells. Legacy `macro_*` keys refer to aggregation over strata, not class-macro F1. Paired contrasts align the same seed/fold/language/holdout cells. Historical 95% intervals use 10,000 seed-cluster bootstrap replicates with RNG seed 250217749; only three seed clusters are available, and folds are not independent clusters. Gate D is descriptive. Current saved modes preserve these verified historical interval bounds rather than rebootstrap them. A5−A4's near-null/negative values are retained; no benefit is manufactured by removing A4 or changing signs.

## Resources and verification evidence

Local cached CLI observations: smoke 2.576 s; audit 56.343 s; Table 2 .133 s; Table 3 .129 s; all-saved .179 s. Initial four-language feature/attack-cache construction takes minutes, but an exact uncached benchmark and peak RAM were not measured. Task 1 raw data occupy approximately 142 MiB; upstream duplicate data, dependencies and caches require additional storage (allow several GB). CPU is sufficient. Full formal training has not been repeated and has no claimed runtime estimate.

The current local suite completed 481 tests passed, 19 warnings, in 109.32 s; [audit/hardened_tests.xml](audit/hardened_tests.xml) records this execution. Run `python scripts/fetch_upstream.py` before either test suite: the release/audit suite includes an actual C smoke fit and also requires raw upstream data. Then run `python -m pytest tests -q` and `python -m pytest repro/2502.17749/v1/tests -q`, or combine them with `python -m pytest tests repro/2502.17749/v1/tests -q`. This local test count is not attributed to the earlier public v1.0.1 release. A one-fold smoke fit provides limited end-to-end evidence and is not a replacement for full training.

## External validation and availability

CodeMirage / Gate E external validation has not started; feasibility assessment is pending the reproduction gate. No external experiment, dataset acquisition or result is asserted here.

Raw LPcode data are available from the pinned upstream source and are not redistributed because no open licence was identified. Frozen numerical evidence and figure sources are publicly inspectable in the [project repository](https://github.com/cdhuangjin/lpcode-mstf-leakage-safe-detection), without an additional open-data reuse grant. MIT applies only to authors' new software under [LICENSE_SCOPE.md](LICENSE_SCOPE.md); see [DATA_LICENSE.md](DATA_LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). 原始数据从固定上游版本获取；本仓库公开的指标与绘图源数据不附加开放复用许可，作者新增软件单独适用 MIT。No CC BY licence, new release version, DOI or archival preservation guarantee is implied.
