# Release status and limits

Version 1.0.0, 2026-09-05. This is an authorised public research-review deposit, not a journal acceptance or a DOI-backed archive.

Included: every research module and test from the frozen implementation; public path/figure-QA helpers; complete primary Gate ledgers and their configurations/manifests; ablation, mechanism, negative-pair sensitivity and figure source data. Eight tiny historical baseline metric pickles are included for manifest tests, not trained models; the fetch script copies them without unpickling. Raw LPcode dataset and original program files are fetched from the pinned owner repository, not republished.

Public-release changes are limited to portable manifest reads, package discovery, creation of the pytest temporary parent, and replacement of a private figure QA dependency with a disclosed narrow public geometry validator. Feature extraction, splits, model construction and formal result bytes remain unchanged. SOURCE_INVENTORY records original and release hashes.

Not included: credentials, virtual environments, duplicate archives, machine-local authoring prompts, third-party PDFs/template class files, training caches and model checkpoints. High-resolution TIFFs can be regenerated; vector PDF/SVG and PNG are included.

## Validation record

Fresh Python 3.11 environment installed from pinned requirements. Installation failures exposed and corrected automatic package discovery and missing pytest temporary-directory setup. Early adapter experiments were rejected because their source hashes would break frozen contracts; the final public feature implementation is the unchanged historical code. No safety checks or immutable bindings were disabled to make tests pass.

The standalone verifier checks 20 frozen Gate files, Gate record counts 480/960/1440/48 and 1800 ablation records, reproducing +1.683/+7.563/+11.160/+7.141 pp from saved evidence without fitting. Five revised figures regenerate without an author-local skill installation. Complete research tests: 423 passed, 19 warnings, 81.33 seconds; separate release checks: 4 passed. The warnings are Matplotlib deprecations and the existing small-fixture MLP convergence warnings. TIFF export uses lossless LZW compression after an uncompressed-export disk-write failure; no figure values or assertions were removed. Full formal experiments were not rerun.

## Remaining rights/preservation conditions

The author has authorised public posting, but has not yet confirmed a permissive code/data licence. `LICENSE` therefore retains rights explicitly instead of inventing MIT/CC BY approval. Publication availability and permission for unrestricted reuse are different claims. An open-licence decision can be applied in a subsequent version after confirmation, with inherited LPcode rights handled separately.

No Zenodo/DataCite DOI is claimed. A versioned GitHub release is inspectable but is not an independent long-term preservation guarantee. Source attribution and observed publication-version differences are detailed in `release_refs_01_11.md` and `release_refs_12_21.md`; single-source checks are not misrepresented as independent multi-source confirmation.
