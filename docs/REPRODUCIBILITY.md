# Reproducibility levels and command map

## Level 1: inspect and verify the reported evidence (tested, no fitting)

From the repository root after README installation:

```bash
python scripts/verify_release.py
python scripts/quickstart.py
python -m pytest tests -q
python tools/revision_figures.py
```

The first command checks original registry file bytes, all four Gate PASS states, 480/960/1440/48 record counts, 1800 ablation records and headline arithmetic. Preserved summaries are research outputs, not a replacement for raw-data/model replication. The complete research tests additionally need `python scripts/fetch_upstream.py` before `python -m pytest repro/2502.17749/v1/tests -q`.

## Level 2: rerun training (source/commands supplied; NOT rerun for this release)

Do this only in a separate rerun checkout, with published historical `results` moved aside to a backup, not deleted or mixed with newly fitted ledgers. Likewise move aside `repro/2502.17749/v1/results`. The runners enforce canonical Gate locations and immutable implementation hashes; an arbitrary output directory cannot be substituted everywhere. Install and fetch upstream there using the README.

```bash
python -m lpcode_v1.t1_strict
python -m lpcode_v1.gates --output-root results/01_transition_test_strict_origins
python -m lpcode_v1.t3
python -m lpcode_v1.t4
python -m lpcode_v1.t5
```

Gate A is strict-origin original10; t3/t4/t5 are held-out generator, deterministic transformations and held-out language. Their main functions publish validated summaries after completion. Gates B–D require the preceding strict PASS artifacts. Running the original, non-strict `t1` is not a substitute for Gate A.

To reconstruct the historical upstream baseline manifest, run the upstream `experiment/task1/main.py` and `experiment/task2/main.py` from their respective directories with `--lang c`, `cpp`, `java`, `py`, `--seed 42 --k 5`. These produce the eight original baseline output files expected by `lpcode_v1.manifest`. Do not execute untrusted pickle files; these are generated locally by the pinned program.

Then build a **new** registry and follow-on analyses in that rerun checkout:

```bash
python -m lpcode_v1.canonical --cross-language-root repro/2502.17749/v1/results/04_cross_language --write-state
python -m lpcode_v1.negative_pair_robustness --output-root results/negative_pair_robustness
python -m lpcode_v1.ablation run --registry results/06_paper_assets/frozen_result_registry.json --output-root results/05_mechanism_analysis
python -m lpcode_v1.mechanism attack --registry results/06_paper_assets/frozen_result_registry.json --t4-root results/03_style_attack
python -m lpcode_v1.mechanism importance --registry results/06_paper_assets/frozen_result_registry.json
```

Run `--help` on each module for cache/output overrides. Ablation reuses complete Gate A/B caches and saved split contracts; importance analysis may fit models, so it is not part of the no-fit verifier. Do not run these commands against the deposited frozen ledgers. Negative-pair analysis in this study is original10 A1 versus A0.

## Parameters, resource and determinism limits

Seeds 42/123/2024; five grouped folds where applicable; four languages/four generators; XGBoost 300 estimators, depth4, learning_rate.05, subsample.8, colsample_bytree.8, n_jobs1. See actual frozen config files for the full contract. Gate D is not five-fold. Task1 raw files total approximately142 MiB; the upstream repository includes duplicate data and Task2. Allow several GB for dependencies, caches and temporary outputs. GPU/API costs are not required. Full matrix wall time and peak memory on a new machine were not measured; no invented timing guarantee is supplied.

The public original-feature compatibility implementation retains the historical source digest and is checked against pinned upstream functions. Packaging/portable-reader/figure-QA adaptations are listed in SOURCE_INVENTORY; training, split, feature and model bodies are unchanged. Do not claim byte-for-byte retraining replication from passing saved-result checks; newly fitted outputs must be audited independently.

## Manuscript tools

`tools/revision_figures.py` is the portable revised-figure entry point. Other `revision_*` tools document authoring provenance and may require author-side originals, Pandoc, OfficeCLI, PyMuPDF, a Wiley template and/or optional external figure-audit tooling. They are not dependencies for scientific analysis. Do not label the document-production workflow as a clean-room portable LaTeX build when the third-party template is not redistributed.
