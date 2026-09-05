# Reproducibility gate before external work

Status: PASS, 5 September 2026. CodeMirage acquisition and evaluation may now
proceed to feasibility review; this is not evidence of an external result.

| Required check | Evidence | Status |
| --- | --- | --- |
| Complete tests | hardened_tests.xml: 481 tests, no failures/errors/skips; 109.32 s pytest elapsed | PASS |
| Headline arithmetic | 58 checks recomputed from frozen cell ledgers | PASS |
| Manifest and hash bindings | Updated 221-file release input inventory, pinned upstream and reconstructed caches | PASS |
| Dual-endpoint isolation | All 2,928 A–D records reconstructed; combined endpoint intersections zero | PASS |
| Exact-content isolation | Raw endpoint and successful transformed-test intersections zero | PASS |
| Matched ablation manifests | 1,800 A0–A5 records bind to formal A/B pair hashes | PASS |
| Actual smoke fit | C, seed 42, fold 0/5; fixed A0/A1 models fitted and evaluated | PASS |
| Saved-table regeneration | Tables 2–7 CSV/Markdown and eight figure-source CSVs | PASS |
| Reviewer documentation | README and reproducibility.md independently reviewed; commands and links checked | PASS |

Final combined audit: `recomputed/repro-gate-final/report.json`, exit 0,
64.337 seconds with existing validated caches. Its manuscript and isolation
subreports both report PASS. Run artifact hashes are stored beside it.

The earlier public v1.0.1 inventory is preserved byte-for-byte in
`v1.0.1_file_manifest.json`. Among its listed files, only README and ignore
rules changed. Existing research implementations, frozen evidence and
manuscript files retain their original bytes. New scripts and documentation
are separately included in the current inventory; generated audit reports
use run-specific artifact hashes rather than circular self-referential hashes.

Boundary: this gate verifies saved evidence, independently reconstructed
raw-data/cached-feature contracts and one fitted smoke fold. It does not
claim full independent retraining of historical Gate A–D. Historical source
archives are byte-attested separately from current implementation equality.
Gate D origin identities remain language-scoped; exact-code comparisons are
global across languages. Existing three-seed statistical limits remain.
