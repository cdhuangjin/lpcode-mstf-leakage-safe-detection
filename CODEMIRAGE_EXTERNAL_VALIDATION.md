# CodeMirage external validation
**Status: feasibility audited; E1/E2 BLOCKED and NOT RUN.**

The reproducibility prerequisite passed before the dataset audit. The [official public release](https://huggingface.co/datasets/HanxiGuo/CodeMirage/tree/174c25ffa9e9f218f8526d403b3aba9631dcfc43) contains only four CSV fields: code, language, source and variant. Neither the schema nor the listed repository files provides human-origin or direct-parent mappings.

Q1 (actual human-source/paraphrase pair) is not established; Q2 (verified nonmatching candidate) and Q3 (candidate human origin) cannot be certified. A source/model label is not an origin key. Furthermore, the [paper](https://arxiv.org/html/2506.11059v1) describes paraphrasing generated code after summary-conditioned generation. This requires a task-semantics check, not just a missing-column repair.

| Item | Result |
| --- | --- |
| Dataset rows reported by official APIs | 209,988 |
| Rows in supported language intersection | 83,999; not verified pairs |
| Actual paired examples established | Not established |
| A0/A4/A5 fitted on CodeMirage | No |
| F1 / precision / recall / AUROC / MCC / CI | NA — not measured |
| Negative, mixed or positive external effect | None can be inferred |
| Raw snippets redistributed | No |
| Pair labels inferred from row order | No |

Detailed evidence: [dataset audit](external/codemirage_dataset_audit.md), [schema probe](external/schema_probe_notes.md), [source metadata](external/codemirage_source_metadata.json). The licence metadata is CC-BY-NC-ND-4.0; no broader permission is inferred.

No external scores, predictions or performance plot were produced. Request an author-verified versioned ancestry mapping and clarify direct-human paraphrase availability. The [request draft](external/provenance_request_draft.md) has not been sent. Existing results remain limited to LPcode.
