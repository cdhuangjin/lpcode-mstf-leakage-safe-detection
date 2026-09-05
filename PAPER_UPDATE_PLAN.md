# Paper update plan
**Ready-to-use text, not a claim that the frozen Word/PDF was recompiled.**

Current frozen paper: [paper_revised.md](results/07_manuscript/paper_revised.md), with associated Word/PDF in the same folder. Its bytes were preserved to maintain the manuscript audit contract.

1. Replace the outdated pending-author-licence statement with the confirmed MIT scope for author-new software. Keep third-party, raw-data, manuscript and figure rights separate.
2. Update reproducibility reporting: 423 original research tests; 427 original total tests; 481 hardened total tests. Add verified raw isolation, reviewer entry points and limited one-fold training, without claiming full training replication.
3. Retain the explicit absence of independent external validation. The CodeMirage eligibility audit belongs in limitations or a reproducibility appendix, not a scored Results subsection.
4. Preserve A5's frozen 112-dimensional definition and near-null relative-block result. A4 is the simpler 84-dimensional core representation, not a retrospectively selected replacement.
5. Do not add an empty or fabricated external-effect figure or efficacy table. The status table provided here is for handoff, not a main-text performance table.
6. After publication of this hardening commit, cite its exact Git revision if citing these new additions. The existing v1.0.1 tag remains valid for its original scope; do not call a moving branch an archival DOI.

## Draft files
- [Reproducibility and licensing replacement](paper_update/reproducibility_update.md)
- [Eligibility methods](paper_update/gate_e_methods.md)
- [Results boundary](paper_update/gate_e_results.md)
- [Discussion](paper_update/gate_e_discussion.md)
- [Limitations](paper_update/gate_e_limitations.md)
- [Gate E status table](paper_update/table_gate_e.md)
- [Figure disposition](paper_update/figure_caption_gate_e.md)
- [A4/A5 positioning](paper_update/a4_a5_positioning.md)

## Not completed
No external Methods text describing executed training, new scores, error-case attribution or external generalisation conclusion is warranted. External adapter, five-seed run, result/failure scripts and plot remain blocked. An original LPcode per-pair failure analysis is separate work: no Gate E predictions exist, and aggregate importance cannot establish case-level errors.

This handoff does not alter the submission template or compile a new DOCX/PDF. Integrate the supplied replacement prose in the next manuscript revision, then rerun numerical and layout checks.
