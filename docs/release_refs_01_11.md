# Release bibliography audit: references 1–11

Checked 2026-09-05 against `results/07_manuscript/revised_sources/bibliography.json`. Read-only audit; no manuscript/bibliography records edited. `nature-ref-verifier` and its common-patterns reference were followed. All eleven stored DOI arrays are empty, so there was no existing DOI to resolve initially. For subsequently discovered publication DOIs, Crossref was requested first, successfully by PowerShell HTTPS when browser retrieval of the API failed.

## Outcome and limits

No fabricated paper, author-order error, or DOI-to-unrelated-paper mismatch was found. However, bibliographic existence is not equivalent to a fully current publication citation. References 1, 5, 6, 7, 8 and 11 have verified published counterparts; 4 needs explicit version handling. Do not label all eleven as independently multi-source verified: 3 and 4 were confirmed from arXiv only; 9 from USENIX only. A landing page and its BibTeX are one source, not two independent sources. Metadata verification does not itself establish every manuscript claim supported by a reference.

## Field-level findings

| Ref | Title and authors/order | Year / venue / location | DOI and source coverage | Finding |
|---|---|---|---|---|
| 1 | Four authors match: Park, Jin, Cha, Han. Stored title matches arXiv; journal title is **Detecting code paraphrased by large language models using coding style features**. | arXiv first submitted 25 Feb 2025; current v3 dated 9 Jan 2026. Published EAAI **162**, article **112454**, 2025. | Crossref 200 for `10.1016/j.engappai.2025.112454`; arXiv, publisher-indexed record and official author repository agree. | Check suggested: prefer published citation with its correct title; if intentionally citing the reproduced preprint, pin the actual consulted version. Do not attach journal DOI to the old title without explaining version relationship. |
| 2 | Title and all eight authors in stored order match USENIX and arXiv. | USENIX Security 2022, **3971–3988**, matches official BibTeX. arXiv first-upload 2020 is not the conference year. | No DOI claimed; USENIX plus arXiv. | Verified across sources. Keep conference year 2022. |
| 3 | Title and five authors match arXiv. | Initial date **27 May 2025** despite identifier beginning 2506; do not infer June from identifier. Stored year 2025 correct. No publication venue established by this audit. | arXiv v1 only; no independently verified published DOI. | Single-source confirmed, not multi-source Verified. Keep explicit arXiv citation. |
| 4 | Four authors and the current long title match arXiv v2. | First upload 2 Sep 2024; current v2 **22 Dec 2025**, submitted to journal, not confirmed published. | arXiv only; no published DOI established. | Check suggested: title/current-content vs initial-year ambiguity. Pin `2409.01382v2` and distinguish 2025 revision from 2024 original upload, especially when describing contemporary models. |
| 5 | Title and three authors match Crossref and arXiv. ArXiv's machine-generated phrase “and 7 other authors” is a parser artifact; actual author heading and Crossref list exactly three. | Published **SANER 2025**, **394–405**. Stored 2024 is preprint year only. | Crossref 200, `10.1109/SANER64311.2025.00044`; arXiv explicitly gives SANER 2025 journal-reference field. | Check suggested: upgrade venue/year/DOI together; do not mix preprint year and conference metadata. |
| 6 | Title and all five authors match arXiv and PMLR. | Published ICML 2023, **PMLR 202:24950–24962**. Stored 2023 correct. | PMLR official proceedings plus arXiv. No DOI needed or invented. | Check suggested: use published PMLR citation rather than arXiv-only venue. |
| 7 | Title and five authors/order match arXiv and official accepted PDF. | Published **ICLR 2024**; stored 2023 is preprint year. | Official OpenReview PDF `Bpcgcr8E8Z` plus arXiv. Forum page browser challenge; PDF accessible/indexed. | Check suggested: upgrade year/venue together. Do not invent volume/pages/DOI. |
| 8 | Five authors/order agree. Stored short title matches arXiv. TMLR PDF adds subtitle **Stress Testing AI Text Detectors Under Various Attacks**. | Published **Transactions on Machine Learning Research, January 2025**; stored 2023 is preprint year. | Official TMLR/OpenReview PDF plus arXiv; forum access failed. | Check suggested: decide cited version, then update title/year/venue consistently; do not add fabricated pages or DOI. |
| 9 | Title and all seven authors/order match USENIX. | USENIX Security **2015:255–270**, exact official BibTeX match. | USENIX landing page/BibTeX only in this audit, no DOI claimed. | Primary-source confirmed; do not overstate independent multi-source coverage. |
| 10 | Title and five authors/order match Crossref and official ESEC/FSE page; Rebryk is present. | ESEC/FSE **2021**, Crossref pages **932–944**; stored year/venue correct but pages omitted. | Crossref 200, `10.1145/3468264.3468606`; official conference and author-hosted camera-ready PDF. | Verified across sources; optional completeness update: pages and DOI. Ignore stale secondary arXiv indexing that omits Rebryk. |
| 11 | Title and all eleven authors/order match Crossref, ACL Anthology and arXiv. | Published **Findings of ACL: EMNLP 2020**, **1536–1547**. Stored year correct. | Crossref 200, `10.18653/v1/2020.findings-emnlp.139`; ACL plus arXiv. | Check suggested: replace arXiv-only venue with published venue/pages/DOI. |

## Source ledger

- [1 arXiv](https://arxiv.org/abs/2502.17749), [1 Crossref](https://api.crossref.org/works/10.1016/j.engappai.2025.112454), [1 publisher](https://www.sciencedirect.com/science/article/pii/S0952197625024856), [1 official author repository](https://github.com/Shinwoo-Park/LPcode). Publisher direct fetch returned 403; metadata also available from publisher search-index result, Crossref and author repository, not silently treated as a successful full-text fetch.
- [2 USENIX](https://www.usenix.org/conference/usenixsecurity22/presentation/arp), [2 arXiv](https://arxiv.org/abs/2010.09470).
- [3 arXiv](https://arxiv.org/abs/2506.11059).
- [4 arXiv v2](https://arxiv.org/abs/2409.01382v2).
- [5 Crossref](https://api.crossref.org/works/10.1109/SANER64311.2025.00044), [5 arXiv](https://arxiv.org/abs/2412.14611). [Author replication deposit](https://zenodo.org/records/13908858) has an earlier title; its Zenodo DOI is an artifact DOI, not the IEEE paper DOI.
- [6 PMLR](https://proceedings.mlr.press/v202/mitchell23a.html), [6 arXiv](https://arxiv.org/abs/2301.11305).
- [7 accepted PDF](https://openreview.net/pdf?id=Bpcgcr8E8Z), [7 arXiv](https://arxiv.org/abs/2310.05130).
- [8 TMLR PDF](https://openreview.net/pdf/a06fe5303ab3d0c635e565d4eae91e5ab7ac8175.pdf), [8 arXiv](https://arxiv.org/abs/2303.11156).
- [9 USENIX](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/caliskan-islam).
- [10 Crossref](https://api.crossref.org/works/10.1145/3468264.3468606), [10 official conference](https://2021.esec-fse.org/details/fse-2021-papers/75/Authorship-Attribution-of-Source-Code-A-Language-Agnostic-Approach-and-Applicability), [10 author-hosted camera-ready](https://sback.it/publications/fse2021.pdf).
- [11 Crossref](https://api.crossref.org/works/10.18653/v1/2020.findings-emnlp.139), [11 ACL Anthology](https://aclanthology.org/2020.findings-emnlp.139/), [11 arXiv](https://arxiv.org/abs/2002.08155).

## Release action recommendation

These findings do not block uploading reproducibility code. They should inform an accurately qualified bibliography audit and subsequent manuscript metadata synchronization. No Nature-only venue filter was used to discard foundational CS works. No formal experiment or frozen numerical record was changed.
