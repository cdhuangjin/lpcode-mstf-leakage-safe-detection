# Data access, dictionary and provenance

## Routes

| Material | Access route | Location |
| --- | --- | --- |
| Raw human/candidate snippets | Reused third-party public source, not relicensed here | Upstream LPcode commit b3660c8262ae57e14498528119607ee673d4257a, `experiment/task1/dataset/{c,cpp,java,py}.jsonl` |
| Historical formal records/configuration | This versioned research deposit | `results/01_transition_test_strict_origins`, `02_unseen_llm`, `03_style_attack`; Gate D at `repro/2502.17749/v1/results/04_cross_language` |
| A0–A5 ablation and decomposition | This deposit | `results/05_mechanism_analysis` |
| Negative-pair sensitivity | This deposit | `results/negative_pair_robustness` |
| Figure source data | This deposit | `results/07_manuscript/revised_sources/evidence.json`, mechanism CSVs and summaries |
| Raw model predictions/checkpoints | Not deposited | No prediction-level case study or pretrained checkpoint is claimed |

Fetch upstream with `python scripts/fetch_upstream.py`. Task 1 records per language are C=3656, C++=3080, Java=11952, Python=15480; classes are balanced in each source file. The upstream dataset copies under `lpcode_dataset` and `experiment/*/dataset` were identical in the original audit. The fetch script pins the actual commit; `docs/UPSTREAM_CHECKSUMS.json` contains exact dataset SHA-256 and sizes to check downloads.

## Numerical dictionary

- `f1`: positive-class F1 in [0,1]; `precision`, `recall`, `accuracy` likewise dimensionless.
- `seed`: cluster identifier, one of 42, 123, 2024. `fold`: dependent partition within seed where applicable; not an independent dataset.
- `language`, `heldout_llm`, `heldout_language`, `condition`: evaluation strata.
- `method`, `representation`, `model`: explicit model/representation identifiers; the negative-pair field named `mstf` denotes original10 A1, not enhanced112 full MSTF.
- `mean_delta_f1` and analogous delta fields: differences on [0,1] scale; multiply by 100 for percentage points.
- `ci_95.low/high`: percentile seed-cluster interval bounds; not SD, not prediction intervals, not a guarantee of population coverage with only three seeds.
- `*_sha256`, `config_id`, `split_hash`: provenance/integrity bindings, not outcome measurements.
- Legacy `macro_*` keys mean equal-weight averages across experiment strata, not F1 averaged over class labels. See manuscript definition.
- Missing fields mean that quantity was not supplied for that record; do not fill them with zero. Original formats/values are preserved.

## Figure/table mapping

Figure 1 / Table 1: features and representation source. Figure 2: frozen Gate protocol/configuration. Figure 3 / Table 2: Gate summaries. Table 3 / Figure 4 left: ablation summaries; Figure 4 right: grouped mechanism summary. Table 4 and Table 6 / Figure 5 right: per-generator/per-language scores. Table 5 / Figure 5 left: attack decomposition. Table 7: negative-pair summary.

Files use JSON, JSONL and CSV. Frozen files keep their original byte content, including historical author-machine paths. `release_paths.py` maps only the documented original prefix to this checkout at read time, so hashes stay unchanged. No credentials or private raw data are required for no-fit result verification.

## FAIR limits

Versioned GitHub tag/release and file checksums provide a stable inspectable route; there is no claimed archival DOI or preservation guarantee. Licence confirmation for original contributions remains explicitly recorded, and upstream reuse rights remain separate. Do not describe this as fully FAIR/openly licensed until those conditions are satisfied.
