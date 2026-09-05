# Submission hardening implementation plan

> For agentic workers: follow the supplied phase order. Use subagent-driven
> development for bounded implementation and independent review; never start
> external training before the reproducibility gate passes.

**Goal:** independently verify the public deposit, add reviewer entry points,
then evaluate CodeMirage only if provenance and reproducibility permit it.

**Architecture:** preserve nested research modules and frozen bytes. Add
scripts, tests and separate audit/recomputed outputs. Reuse the existing
extractors, pairing, fixed models and ledger contracts.

**Tech stack:** pinned Python 3.11 environment, pytest, existing NumPy /
scikit-learn / XGBoost / tree-sitter code; GitHub versioned sources.

- [x] Phase 0: inspect repository contracts and write repository_inventory.md.
- [x] Phase 1: new clone + venv, pinned install, upstream acquisition, all tests
  with JUnit XML; report exact counts, environment and elapsed time.
- [x] Phase 2: test-first saved-ledger aggregation, strict matched contrasts,
  rounded manuscript checks; JSON/Markdown audit, fail on missing cells.
- [x] Phase 3: verify hashes, reconstruct pair isolation from raw sources and
  compare saved split digests; missing evidence is BLOCKED, never zero.
- [x] Phase 4: test-first reproduce.py modes smoke/audit/table2/table3/all-saved;
  smoke must actually fit fixed A0/A1 on one held-out fold.
- [x] Phases 5–6: README and reproducibility.md with tested commands, output
  mapping, measured runtime and frozen/recomputed/full-training distinctions.
- [x] Reproducibility gate: all required checks PASS before external work.
- [x] Phases 7–9: official CodeMirage source/licence/schema and eligibility audit; Q1/Q3 provenance is not established. See external/codemirage_dataset_audit.md. E1/E2 are BLOCKED, not negative results.
- [ ] Phases 10–14 (BLOCKED): adapter and configuration require verified
  positive and negative provenance feasibility; freeze E1 five-seed A0/A4/A5
  configuration before scoring. No parser or hyperparameter expansion.
- [ ] Phases 15–21 (BLOCKED): run shared manifests, save all metrics and hashes, inspect
  errors only from saved predictions, generate paired tables and simple plot.
- [x] Phases 22–27, 29–30: boundary-aware manuscript draft additions, A4/A5 positioning, CodeBERT feasibility only, final reports and HOLD recommendation. Frozen Word/PDF remains unchanged; no external-result prose is invented.
- [ ] Phase 28 external artifact CSV utility (DEFERRED with the blocked run); existing smoke/saved-output SHA-256 manifests are complete.

The supplied task document defines the full requirements. Later implementation
details depend on audited schemas; no adapter or experimental labels will be
invented ahead of that audit. Report any unresolved blocking condition with
the precise next decision required.
