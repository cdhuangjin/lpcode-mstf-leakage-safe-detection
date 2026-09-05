# Statistical reporting and source-data audit

Scope: saved run-level evidence only; no formal refitting or new significance tests during release. Exact code metric is sklearn positive-class F1. Equal weighting is across the saved language/holdout cells, not across class labels.

Three seeds (42,123,2024) are resampling clusters. Paired bootstrap draws retain all dependent folds and applicable language/generator strata inside the seed. Saved routines use 10,000 replicates. Treating the 480/960/1440/48 records as independent sample counts would be incorrect. Gate D has no five-fold split and remains descriptive.

All principal contrasts include absolute scores, differences and defined uncertainty. Main Gate A is A1−A0 under a fixed XGBoost/original10 contract; B–D compare the full representation/XGBoost system with original10/MLP. Larger shifted-setting differences cannot identify a controlled transition-by-environment interaction. C1–C5 are incremental/composite comparisons as named in the ablation table. C4 clean −0.019 pp and unseen +0.007 pp are retained.

No multiplicity-adjusted hypothesis-test family or p-value claims are introduced. The displayed bootstrap intervals are not simultaneous familywise intervals. Exploratory feature ranks and environmental breakdowns are descriptive; ranking is not causal attribution or an independent ablation. Small-cluster and single-corpus limitations remain.

Combined transformation excludes comment injection. Absolute clean-to-combined F1 drops are 6.311 and 1.935 pp; 69.33% denotes the relative reduction of those absolute drops, not clean-normalised accuracy or an additional detection metric.

Audit severity: no newly identified P0 numerical inconsistency. Remaining P1 inferential limits are three seed clusters, one corpus and restricted generator/language/transformation coverage. Missing prediction-level FP/FN records prevent a supported case-pair analysis. These limits are not hidden by repository publication.
