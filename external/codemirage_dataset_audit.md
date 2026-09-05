# CodeMirage eligibility audit
Date: 2026-09-05. Status: **BLOCKED for the frozen paired-provenance task**.

## Source and rights
Official dataset: https://huggingface.co/datasets/HanxiGuo/CodeMirage
Pinned revision: `174c25ffa9e9f218f8526d403b3aba9631dcfc43`.
The card declares CC-BY-NC-ND-4.0. This records metadata, not a legal conclusion permitting adaptation or redistribution. No raw corpus or snippets are deposited here. Author-new software MIT does not relicense this dataset.

See [schema probe](schema_probe_notes.md) and [observed metadata](codemirage_source_metadata.json) for API URLs, file sizes, publisher LFS identifiers and counts. Both actual CSV headers were checked by 128-byte HTTP Range requests (206); full CSV bytes were not downloaded or locally SHA-256 verified.

## Schema and populations
Both CSVs contain only `code,language,source,variant`. There is one code field per row, not separate source/candidate fields. `source` names Human or a generator, not a human-origin ID. No program, parent, pair, repository, prompt or origin ID is exposed. The pinned tree lists only the card, attributes and two CSVs; no mapping sidecar is listed.

Official server statistics, observed on the audit date, report 146,992 train and 62,996 test rows (209,988 total), ten languages and ten generators. The supported intersection C/CPP/Java/Python contains 83,999 rows, **not 83,999 verified pairs**. There are two null-code entries in train. These are unpinned server statistics; file identities and limitations are recorded in the probe.

Duplicate handling: not performed on the full corpus. Missing-code filtering, origin components and global exact-code hashing would be required after acquisition and provenance resolution. Dataset train/test labels alone do not certify our isolation contract.

## Eligibility questions
| Question | Finding |
| --- | --- |
| Q1: human source and its actual paraphrase? | Not established by the public schema. |
| Q2: source and a nonmatching candidate? | Cannot verify nonmatching origin without Q1/provenance; arbitrary pairing is not accepted ground truth. |
| Q3: negative candidate's original human origin? | Missing. Dual-endpoint isolation cannot be asserted. |

The [official paper](https://arxiv.org/html/2506.11059v1), section 3.1 and Appendix D, describes a human-code summary used for generation, followed by paraphrasing of generated code. This differs from directly paraphrasing a supplied human endpoint. Even a recovered mapping therefore needs a task-compatibility review.

Row order, equal category counts, viewer row_idx and code similarity are not provenance evidence. A relaxation of Q3 alone cannot fix missing Q1 or incompatible labels.

## Decision and restart requirements
E1 and optional E2 were not run; no pair labels, adapter outputs, predictions, failure cases, F1, confidence intervals or effect plot exist. This is an eligibility failure, not a negative model result. Planned A0/A4/A5 and five seeds (42, 123, 2024, 3407, 7777) remain conditional, not an executed configuration.

Obtain an author-supplied mapping binding candidate IDs/hashes to direct parent IDs/hashes, ultimate human origins, language, generator, transformation stage and dataset revision. Ask whether direct human-to-paraphrase examples exist. Verify mapping coverage and rights, then determine compatibility before freezing a run. No fallback to single-endpoint AI-code classification is authorized.

No provenance request was sent during this audit. The internal correspondence draft is not distributed in the current branch.
