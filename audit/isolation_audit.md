# Phase 3 independent isolation and provenance audit

Status: PASS on 5 September 2026. Machine-readable authority:
`isolation_audit.json`, produced by `scripts/audit_isolation.py`.

The first audit built feature and transformation caches from the pinned
upstream data in the fresh public checkout. No author-side caches or models
were copied. The final run reused these validated caches; its JSON discloses
pre-existing cache files. A repeated audit is not a new feature-extraction run
when a valid cache is available. No model was fitted in this phase.

| Gate | Matched frozen records | Reconstructed split cells | Combined endpoint overlap | Exact-code overlap |
| --- | ---: | ---: | ---: | ---: |
| A | 480 / 480 | 60 | 0 | 0 |
| B | 960 / 960 | 240 | 0 | 0 |
| C, before transformation | 1,440 / 1,440 | 60 | 0 | 0 |
| D | 48 / 48 | 12 | 0 | 0 |

Both endpoint roles are combined before train/test intersection, including
cross-role collisions. For Gate D, origin IDs are language-scoped: this does
not prove separation of latent semantic tasks across languages. Exact UTF-8
code hashes are compared across all languages without language scoping.

Additional checks:

- All 1,800 A0–A5 ablation records bind to formal A/B train/test hashes and
  row counts, with complete expected cell coverage.
- All four raw dataset hashes, recomputed feature-cache semantic hashes,
  configurations, frozen ledgers, registry files and upstream tracked-tree
  bindings match their recorded contracts.
- Gate C's five transformation caches match frozen semantic digests. All
  1,440 clean/transformation ledger records match reconstructed success/output
  sets and counts. Successful transformed test endpoints have zero exact-code
  intersections with either unchanged training endpoint role.
- A missing or malformed input generates a fresh structured FAIL report;
  unavailable overlap counts remain null, not zero.

## Historical source version boundary

The current `t3.py` is not byte-identical to the versions used by the frozen
gates because it contains the subsequent negative-pair extension. This remains
explicitly recorded as current-source inequality. Two recovered archival
versions exactly match the historical configuration hashes; see
`historical_sources/README.md`. Independently reconstructed default-protocol
pair digests match the original ledgers. Neither this agreement nor archive
hash equality establishes independent model-training replication.

Do not overwrite the live package with these archives or resume deposited
ledgers in place. Use a separate version-matched tree and new outputs for
historical retraining.

## Verification

Command: `python scripts/audit_isolation.py` (exit 0).

Focused tests: `python -m pytest tests/test_isolation_audit.py -q`:
22 passed; the controller rerun took 6.30 seconds with an explicit local
pytest temporary directory. Independent specification and quality reviews
approved the implementation after the stale-report failure path was fixed.

This completes Phase 3 only. The overall reproducibility gate additionally
requires a genuinely fitted smoke experiment and the reviewer entry points.
