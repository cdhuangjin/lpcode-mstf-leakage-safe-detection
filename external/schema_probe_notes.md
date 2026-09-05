# CodeMirage public schema probe

Observed 2026-09-05. Metadata-only report; no corpus or code snippets saved. Official HTTP API responses were inspected using PowerShell. CSV requests used HTTP Range bytes 0-127 and returned 206 for both files.

## Revision and schema

Pinned repository revision: `174c25ffa9e9f218f8526d403b3aba9631dcfc43`.

- Repository metadata: https://huggingface.co/api/datasets/HanxiGuo/CodeMirage/revision/174c25ffa9e9f218f8526d403b3aba9631dcfc43
- Pinned file inventory: https://huggingface.co/api/datasets/HanxiGuo/CodeMirage/tree/174c25ffa9e9f218f8526d403b3aba9631dcfc43?recursive=true
- Pinned card: https://huggingface.co/datasets/HanxiGuo/CodeMirage/raw/174c25ffa9e9f218f8526d403b3aba9631dcfc43/README.md
- Both actual CSV headers: `code,language,source,variant`.
- Header URLs: https://huggingface.co/datasets/HanxiGuo/CodeMirage/resolve/174c25ffa9e9f218f8526d403b3aba9631dcfc43/train.csv and https://huggingface.co/datasets/HanxiGuo/CodeMirage/resolve/174c25ffa9e9f218f8526d403b3aba9631dcfc43/test.csv
- All four features have string dtype in the datasets-server schema. No origin ID, candidate ID, repository URL, task ID, generation-parent ID, summary, prompt, or positive-pair link is exposed as a column.
- Repository inventory contains only `.gitattributes`, `README.md`, `train.csv`, `test.csv`; no additional public mapping file is listed.
- The card describes `source` as Human or model name, not a provenance key. The card supplies no row-order join contract.

The datasets-server info endpoint references CSV revision `1f57d0edac050e98e96991be023fb2902d9a9cb4`. Its two CSV git blob IDs, sizes, and LFS SHA256 identifiers exactly match the pinned revision's inventory. Its README differs. This establishes file-level identity for the CSVs behind the server info; server statistics URLs themselves are unpinned observations.

| File | CSV bytes | Git blob ID | LFS SHA256 |
|---|---:|---|---|
| train.csv | 493199393 | a435994220f3679e4f059ce6726d62557519caf9 | 2c3d105207a6852495b19b3aa2fb5b4458ed1270c36c1f773123edb8605fdbfc |
| test.csv | 210549770 | ba68abbd8c8f57e8e3f73483baf33ebd709528d2 | d6385634ba8fc1c7c945021bdd0789786eab54d7421a50a4009be0b32220d8d6 |

Comparison inventory: https://huggingface.co/api/datasets/HanxiGuo/CodeMirage/tree/1f57d0edac050e98e96991be023fb2902d9a9cb4?recursive=true

## Exact API counts

Info: https://datasets-server.huggingface.co/info?dataset=HanxiGuo%2FCodeMirage

Statistics (both returned `partial: false`):

- https://datasets-server.huggingface.co/statistics?dataset=HanxiGuo%2FCodeMirage&config=default&split=train
- https://datasets-server.huggingface.co/statistics?dataset=HanxiGuo%2FCodeMirage&config=default&split=test

| Split | Rows | Human source | Normal variant | Paraphrased variant | Null variant | Null code |
|---|---:|---:|---:|---:|---:|---:|
| train | 146992 | 7000 | 69996 | 69996 | 7000 | 2 |
| test | 62996 | 3000 | 29999 | 29997 | 3000 | 0 |
| total | 209988 | 10000 | 99995 | 99993 | 10000 | 2 |

These are column marginals, not cross-tabulations or pair counts. The card associates Human with N/A variant; a metadata-only first-rows check observed Human/null on initial train rows.

| Language | Train | Test |
|---|---:|---:|
| C | 14700 | 6300 |
| CPP | 14700 | 6299 |
| CSharp | 14696 | 6299 |
| Go | 14700 | 6300 |
| HTML | 14698 | 6300 |
| Java | 14700 | 6300 |
| JavaScript | 14700 | 6300 |
| PHP | 14698 | 6298 |
| Python | 14700 | 6300 |
| Ruby | 14700 | 6300 |

| Source | Train | Test |
|---|---:|---:|
| Human | 7000 | 3000 |
| claude-3.5-haiku | 14000 | 6000 |
| deepseek-r1 | 14000 | 5998 |
| deepseek-v3 | 14000 | 6000 |
| gemini-2.0-flash | 14000 | 6000 |
| gemini-2.0-flash-thinking-exp | 14000 | 6000 |
| gemini-2.0-pro-exp | 13992 | 5998 |
| gpt-4o-mini | 14000 | 6000 |
| llama3.3-70b | 14000 | 6000 |
| o3-mini | 14000 | 6000 |
| qwen2.5-coder | 14000 | 6000 |

First-rows schema check: https://datasets-server.huggingface.co/first-rows?dataset=HanxiGuo%2FCodeMirage&config=default&split=train . Six initial rows had only the four data keys, language Python, source Human, variant null. API wrapper `row_idx` is a display position, not a dataset provenance field or a documented pairing key.

Parquet metadata: https://datasets-server.huggingface.co/parquet?dataset=HanxiGuo%2FCodeMirage . Lists one converted file per split: train 183735371 bytes; test 78391378 bytes. Neither was downloaded.

## Bounded feasibility conclusion

The public schema supports source/variant classification. It does not directly establish positive pairs, a candidate's human origin, or whether a paraphrase was directly produced from a particular human snippet. Equal counts and row positions do not establish those relationships. No pair labels were constructed. For Q1/Q3 requiring verified origin-to-candidate relationships, this release alone is insufficient without an author-provided mapping or documented provenance recovery procedure. This is a schema/provenance conclusion, not a claim that no underlying pairs exist. Paper semantics and legal interpretation remain separate checks.
