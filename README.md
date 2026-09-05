# LPcode / MSTF: leakage-safe paired provenance detection

Research code and frozen evidence for **Multi-view coding-style transitions for leakage-safe paired detection of LLM-paraphrased code**, by Jin Huang, Qiao Li and Qisen Gao. This is a manuscript/reproducibility release, not an accepted publication or a pretrained service.

![MSTF architecture](results/07_manuscript/revised_figures/figure1.png)

## What is being detected?

Given a human source and a candidate, study a labelled paraphrase relationship. MSTF combines 28-dimensional endpoint descriptors, signed differences and relative changes into 112 dimensions. This is not candidate-only AI-code detection and does not establish historical provenance or semantic equivalence.

## Installation (Python 3.11)

From the repository root, with Git and Python 3.11 installed:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e repro/2502.17749/v1
```

Core and plotting dependencies are pinned. The older `uv.lock` is retained as historical environment provenance; use `requirements.txt` for this release. CPU execution is sufficient; no paid API or GPU is required. Linux/macOS commands are provided for portability, but the release validation platform is Windows/Python 3.11. Do not claim untested platform equivalence.

## Quick start: inspect without fitting a model

```bash
python scripts/quickstart.py
python scripts/verify_release.py
python -m pytest tests -q
```

Quick start uses synthetic endpoint vectors, not synthetic research results. The verifier checks every frozen Gate file and ablation binding and prints the reported differences without fitting anything. Raw third-party data are not needed for this evidence check.

## Get the original data and run the complete tests

```bash
python scripts/fetch_upstream.py
python -m pytest repro/2502.17749/v1/tests -q
```

The original [LPcode repository](https://github.com/Shinwoo-Park/LPcode) is fetched at commit `b3660c8262ae57e14498528119607ee673d4257a`. Its raw dataset and original `main.py` are not republished here or covered by our rights statement. See [data access and checksums](docs/DATA_ACCESS.md) and [third-party notices](THIRD_PARTY_NOTICES.md). The frozen 10-feature compatibility implementation is included unchanged and tested against upstream; no upstream rights are granted by implication.

## Frozen results

F1 below is positive-class F1 averaged equally across evaluation cells, **not class-macro F1**. Differences are percentage points.

| Gate | Comparison | Baseline | Method | Difference |
| --- | --- | ---: | ---: | ---: |
| Strict clean | A1 vs A0, fixed XGBoost, original10 | .9149 | .9317 | +1.683 pp |
| Held-out generator | full MSTF vs original LPcode MLP | .8915 | .9671 | +7.563 pp |
| Combined transformation | full MSTF vs original LPcode MLP | .8415 | .9531 | +11.160 pp |
| Held-out language | full MSTF vs original LPcode MLP | .8938 | .9652 | +7.141 pp |

The comparators differ: this table does not identify a transition-by-shift interaction. Clean A5−A4 is −0.019 pp; held-out-generator A5−A4 is +0.007 pp. Relative normalization has little incremental benefit. Only three seed clusters underpin reported intervals; Gate D is descriptive. Read [statistical scope](docs/STATISTICS.md).

## Reproduce analyses / figures

```bash
python scripts/verify_release.py
python tools/revision_figures.py
```

The five revised figures consume published frozen source data. Public geometry QA is intentionally narrower than the author's archived publication QA; no proprietary/local Codex skill is required. See [reproduction commands](docs/REPRODUCIBILITY.md) for the formal runners and their prerequisites. Never resume published historical ledgers with altered source contracts. Full retraining was **not** rerun during release preparation.

## Repository contents

- `repro/2502.17749/v1/lpcode_v1/`: complete research implementation (features, representations, isolation, classifiers, Gate A–D runners, ablation, mechanism, negative-pair sensitivity and audits).
- `repro/2502.17749/v1/tests/`: complete existing research test suite.
- `results/`: frozen run-level records/configurations/manifests, ablation/negative sensitivity, importance and attack decomposition, manuscript figure data.
- `tools/`: manuscript/evidence/figure construction and metric-analysis utilities. Office document typesetting additionally needs Pandoc, OfficeCLI and a licensed/author-provided Wiley template; it is not needed to reproduce scientific results.
- `SOURCE_INVENTORY.json`: original-to-release source hashes, including the explicit release-only adaptations.
- `docs/`: data dictionary, provenance, statistics, release verification and training boundaries.

Generated model checkpoints and caches are not distributed. No ready-to-use prediction checkpoint is claimed. Author-local prompts, environments, duplicate archives, downloaded third-party PDFs and credentials are not research source code and are excluded.

## Citation and manuscript

Use `CITATION.cff` for this software/evidence release; the paper is a manuscript, without an invented DOI or acceptance. [Main manuscript](results/07_manuscript/LPcode_MSTF_Wiley_Revised.pdf) and [supplement](results/07_manuscript/LPcode_MSTF_Supplementary.pdf) accompany the release. The structured bibliography is also exported as `references.ris`. Cite the original LPcode work and upstream dataset independently when reusing them.

## Rights and access

The authors' new software is licensed under **MIT**, with exclusions in [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Inherited LPcode code (including `features_official.py`) and third-party material are not relicensed. Data, figures and manuscripts receive no additional open-reuse grant; see [DATA_LICENSE.md](DATA_LICENSE.md). Read `LICENSE` and `THIRD_PARTY_NOTICES.md`. No DOI-backed preservation deposit is asserted. See `docs/RELEASE_STATUS.md` for validation scope and `docs/MANUSCRIPT_LICENSE_ADDENDUM.md` for the post-v1.0.0 update.
