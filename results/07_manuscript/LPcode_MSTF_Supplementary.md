# Supplementary information

## Multi-view coding-style transitions for leakage-safe paired detection of LLM-paraphrased code

Jin Huang, Qiao Li and Qisen Gao

All tables derive from saved evidence. No model fitting or new formal experiment was performed during this revision. F1 is positive-class F1; equal-weight means across experimental cells are not class-macro F1.

# S1 Controlled contrast intervals

| Contrast | Definition | Clean (pp) | 95% interval | Held-out (pp) | 95% interval |
| --- | --- | --- | --- | --- | --- |
| C1 | A1 minus A0 | +1.683 | [1.466, 1.932] | +1.153 | [1.068, 1.259] |
| C2 | A2 minus A0 | +4.592 | [4.258, 4.832] | +4.976 | [4.745, 5.100] |
| C3 | A4 minus A1 | +4.092 | [3.923, 4.300] | +4.659 | [4.499, 4.798] |
| C4 | A5 minus A4 | -0.019 | [-0.079, 0.052] | +0.007 | [0.000, 0.013] |
| C5 | A5 minus A0 | +5.756 | [5.360, 6.003] | +5.819 | [5.645, 5.938] |

C1: signed differences with original features; C2: endpoint expansion; C3: endpoint expansion with signed differences; C4: relative block; C5: full representation versus original endpoints. Intervals retain folds inside three seed clusters. A positive interval for the extremely small unseen C4 is not a claim of practical importance.

# S2 Source registry

| Gate | Protocol | Manifest SHA-256 |
| --- | --- | --- |
| gate_a | all-llm-strict-origin-v2 | 593b2044cab542f120190d4f50bccbdcc06ff9c48cde115608f705448dd2fa95 |
| gate_b | leave-one-llm-strict-origin-v1 | 10cdfea1964e78aae624fba017bd99d219899d07e22ed9c10e4c7ef55397c0db |
| gate_c | all-llm-strict-origin-attack-v1 | 6aaa2d77823b517a1b3ac4649cb2799e03cf04d44045307b649526f45534383b |
| gate_d | leave-one-language-strict-origin-v1 | a04cb43cbf527500bd1d4cf30c833f7d126ca25f81352b05cf7ac46b40bf4373 |

Full file digests are supplied in the revision provenance snapshot and frozen registry. SHA-256 values identify files, not evidence of methodological sufficiency by themselves.

# S3 Per-generator and per-language results

| Held-out generator | LPcode original | Full MSTF | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- |
| GPT-3.5 | 0.8978 | 0.9819 | +8.412 | [7.894, 8.768] |
| Gemini-Pro | 0.8837 | 0.9592 | +7.550 | [7.196, 7.840] |
| WizardCoder-33B | 0.9003 | 0.9687 | +6.836 | [6.463, 7.241] |
| DeepSeek-Coder-33B | 0.8842 | 0.9587 | +7.456 | [7.236, 7.839] |

| Held-out language | LPcode original: mean ± SD | MSTF: mean ± SD | Difference (pp) |
| --- | --- | --- | --- |
| C | 0.8922 ± 0.0046 | 0.9656 ± 0.0010 | +7.339 |
| C++ | 0.8976 ± 0.0034 | 0.9723 ± 0.0009 | +7.475 |
| Java | 0.8799 ± 0.0076 | 0.9584 ± 0.0021 | +7.852 |
| Python | 0.9055 ± 0.0017 | 0.9645 ± 0.0014 | +5.897 |

# S4 Negative-pair sensitivity

| Negatives | A0 F1 | A1 F1 | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- |
| current | 0.9149 | 0.9317 | +1.683 | [1.466, 1.932] |
| random | 0.9119 | 0.9300 | +1.805 | [1.647, 1.896] |
| hard | 0.8463 | 0.8824 | +3.605 | [3.431, 3.808] |

These are A1–A0 comparisons with original ten features, not full MSTF. Current, random and hard refer to the registered negative construction only. Positives and isolation constraints remain fixed.

# S5 Transformation uncertainty

| Condition | Method | Drop (pp) | 95% seed-cluster CI (pp) |
| --- | --- | --- | --- |
| combined | lpcode original | 6.311 | [5.822, 6.849] |
| combined | mstf | 1.935 | [1.776, 2.039] |
| comment injection | lpcode original | 0.034 | [-0.058, 0.125] |
| comment injection | mstf | 0.119 | [0.092, 0.142] |
| comment removal | lpcode original | 0.890 | [0.833, 0.979] |
| comment removal | mstf | 0.187 | [0.168, 0.212] |
| format normalization | lpcode original | -0.001 | [-0.004, 0.003] |
| format normalization | mstf | 0.004 | [-0.001, 0.014] |
| identifier rename | lpcode original | 5.124 | [4.754, 5.543] |
| identifier rename | mstf | 1.732 | [1.642, 1.781] |

# S6 Interpretation and access

All five quantitative transformations are retained; combined applies removal, renaming and formatting, without injection. Parsing checks do not prove semantic equivalence. Per-language transformed scores, change fractions and parser-regression statistics remain in the accompanying attack decomposition CSV. Full grouped importance and ranking stability are available in the local mechanism CSVs. These materials are included in the publicly accessible research deposit.

Code, pinned dependencies, frozen records, figure source data and reproduction commands are publicly deposited at <https://github.com/cdhuangjin/lpcode-mstf-leakage-safe-detection/tree/1b6a5b9f7f274b22a718b53219581d5f57a30792>. Anonymous access was verified on 5 September 2026. Raw LPcode data are obtained from its original pinned repository, with checksums supplied in the deposit. Open-reuse licensing remains subject to author confirmation; no archival DOI or independent formal retraining is claimed.
