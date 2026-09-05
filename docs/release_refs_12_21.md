# Release bibliography audit: references 12–21

Checked: 2026-09-05. Input: `results/07_manuscript/revised_sources/bibliography.json`, records numbered 12–21. This is a read-only field audit under `nature-ref-verifier`; no bibliography/manuscript edits were made. It verifies bibliographic identity, not every claim attributed to each work.

## Summary

- No critical wrong-paper DOI, author-order error, invented title, or incorrect page range was found.
- Six records are multi-source verified without a material field discrepancy: 13–18. Records 13, 14 and 18 have optional published-venue upgrades; their explicitly labelled preprint citations are not false journal citations.
- Record 21 is multi-source verified, giving seven multi-source verified records in this batch.
- Record 19 has a resolved version-dependent author-list warning: the manuscript's 16-author journal list is correct and should remain unchanged.
- Records 12 and 20 are primary-source verified but lack a successful independently maintained second full bibliographic record in this bounded audit. They must not be described as fully multi-source verified.
- Crossref was requested first for both DOIs already present in this batch. Browser retrieval failed, but direct `Invoke-RestMethod` to the same public API succeeded (HTTP success, structured work records). Two newly discovered ACL publication DOIs also returned matching Crossref records.

## Field-level findings

| Ref | Title and authors | Year / venue / locator | DOI | Status and action |
|---|---|---|---|---|
| 12 | GraphCodeBERT title and all 18 names, Guo through Zhou, match arXiv, including Shuo Ren second and Shao Kun Deng twelfth. | Cited 2020 is the first preprint year; arXiv reports ICLR 2021 acceptance and latest v4 in 2021. No conference pages are invented. | No DOI asserted in input. | Primary-source verified; optional switch to ICLR 2021 only after retrieving the definitive proceedings record. OpenReview access encountered a browser-verification page and was not counted as corroboration. |
| 13 | CodeT5 title and four-author order Wang, Wang, Joty, Hoi match arXiv and ACL Anthology; spacing in C.H. is immaterial. | 2021 agrees. Published EMNLP record has pp. 8696–8708. | Newly verified publication DOI: 10.18653/v1/2021.emnlp-main.685; Crossref title, year, venue and pages agree. | Multi-source verified. Prefer published EMNLP citation as editorial improvement; current arXiv citation is identifiable and correct. |
| 14 | UniXcoder title and six-author order Guo, Lu, Duan, Wang, Zhou, Yin match arXiv and ACL Anthology. | 2022 agrees. ACL Volume 1: Long Papers, pp. 7212–7225. | Newly verified publication DOI: 10.18653/v1/2022.acl-long.499; Crossref title, venue, year and pages agree. | Multi-source verified. Optional published-venue upgrade. |
| 15 | CodeSearchNet title and five-author order Husain, Wu, Gazit, Allamanis, Brockschmidt match arXiv and the authors' official GitHub citation. | 2019 and arXiv:1909.09436 agree in both records. | No DOI asserted in input; absence is not a failed DOI check. | Multi-source verified (arXiv plus project-maintained citation); no correction. |
| 16 | Quiring, Maier, Rieck and full title agree between USENIX and arXiv. | USENIX Security 2019, pp. 479–496 exactly match official BibTeX. | No DOI asserted. | Multi-source verified. No correction. |
| 17 | Five-author order exactly matches Crossref: Zhen Li; Guenevere (Qian) Chen; Chen Chen; Yayi Zou; Shouhuai Xu. Crossref separates “RoPGen” and the subtitle; their concatenation matches the manuscript title. | ICSE 2022 agrees; Crossref supplies pp. 1906–1918, currently omitted rather than wrong. | Existing 10.1145/3510003.3510181 resolves to the correct work. | Multi-source verified via Crossref and arXiv. Optional page-range completion; do not mark shortened Crossref title a wrong-paper DOI. |
| 18 | Tianqi Chen and Carlos Guestrin agree. Crossref title “XGBoost” plus subtitle “A Scalable Tree Boosting System” matches full input. | 2016 agrees. Published venue is 22nd ACM SIGKDD, pp. 785–794; current citation identifies the arXiv preprint. | Existing 10.1145/2939672.2939785 resolves to correct publication. | Multi-source verified via Crossref and arXiv. Prefer consistent published KDD citation if revising venue fields. |
| 19 | All 16 authors in the input exactly match the JMLR journal page, including Prettenhofer immediately after Blondel. Latest arXiv v4 adds Müller, Nothman and Louppe, making 19 authors. | JMLR 12(85):2825–2830, 2011 matches. ArXiv initial deposit 2012 is not the journal publication year. | No DOI asserted. | Check suggested, resolved: retain the journal's 16-author list and 2011 year. The differing arXiv author list is version drift, not evidence that journal authors were omitted. |
| 20 | Bengio then Grandvalet and title match JMLR HTML and first-page PDF metadata. | JMLR 5:1089–1105, 2004 agrees; HTML adds the month label Sep. | No DOI asserted. | Primary-source verified. HTML plus PDF are two manifestations from one publisher, not independent corroboration. No correction indicated. |
| 21 | Fisher, Rudin, Dominici and full title match JMLR and arXiv. Apostrophe typography is immaterial. | JMLR 20(177):1–81, 2019 matches both JMLR and arXiv journal-reference field. Initial arXiv 2018 is not substituted for 2019. | No DOI asserted. | Multi-source verified. No correction. |

## Sources actually consulted

- 12: [arXiv GraphCodeBERT](https://arxiv.org/abs/2009.08366). OpenReview challenge is an access limitation, not a verified source.
- 13: [arXiv CodeT5](https://arxiv.org/abs/2109.00859), [ACL publication](https://aclanthology.org/2021.emnlp-main.685/), [Crossref record](https://api.crossref.org/works/10.18653/v1/2021.emnlp-main.685).
- 14: [arXiv UniXcoder](https://arxiv.org/abs/2203.03850), [ACL publication](https://aclanthology.org/2022.acl-long.499/), [Crossref record](https://api.crossref.org/works/10.18653/v1/2022.acl-long.499).
- 15: [arXiv CodeSearchNet](https://arxiv.org/abs/1909.09436), [official project citation](https://github.com/github/CodeSearchNet).
- 16: [USENIX publication and BibTeX](https://www.usenix.org/conference/usenixsecurity19/presentation/quiring), [arXiv preprint](https://arxiv.org/abs/1905.12386).
- 17: [Crossref RoPGen record](https://api.crossref.org/works/10.1145/3510003.3510181), [arXiv RoPGen](https://arxiv.org/abs/2202.06043).
- 18: [Crossref XGBoost record](https://api.crossref.org/works/10.1145/2939672.2939785), [arXiv XGBoost](https://arxiv.org/abs/1603.02754).
- 19: [definitive JMLR journal author list](https://www.jmlr.org/papers/v12/pedregosa11a.html), [later arXiv author-list revision](https://arxiv.org/abs/1201.0490).
- 20: [JMLR HTML](https://www.jmlr.org/papers/v5/grandvalet04a.html), [JMLR PDF](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf).
- 21: [JMLR publication](https://jmlr.org/papers/v20/18-760.html), [arXiv with journal reference](https://arxiv.org/abs/1801.01489).

## Interpretation limits

Independent hosting is not independent scientific validation: Crossref mostly reflects publisher deposits and official project citations are author supplied. The source count concerns metadata corroboration only. No citation-count threshold, citation legitimacy assessment, or review of all paper claims is implied. Optional publication upgrades do not block the GitHub code release. This audit does not grant rights to redistribute any cited paper PDF.
