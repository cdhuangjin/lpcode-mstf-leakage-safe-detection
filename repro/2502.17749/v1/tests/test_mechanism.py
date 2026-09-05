"""Read-only analysis contracts for Gate C mechanism artifacts."""

from __future__ import annotations

import pytest

from lpcode_v1.mechanism import (
    MechanismError,
    aggregate_importance,
    decompose_attack_ledger,
    full_mstf_feature_groups,
    jaccard_top_k,
    mstf_feature_names,
)


def _record(language: str, condition: str, method: str, f1: float) -> dict:
    return {
        "language": language,
        "condition": condition,
        "method": method,
        "seed": 42,
        "fold": 0,
        "f1": f1,
        "test_rows": 10,
        "test_candidate_parse_failures": 0,
        "test_human_parse_failures": 0,
    }


def test_attack_decomposition_requires_all_six_conditions() -> None:
    conditions = ("clean", "comment_removal", "identifier_rename", "format_normalization", "comment_injection", "combined")
    methods = ("lpcode_original", "xgb_original", "best_transition", "mstf")
    records = [_record("c", condition, method, 0.9) for condition in conditions for method in methods]

    report = decompose_attack_ledger(records, "a" * 64)

    assert report["conditions"] == conditions
    assert len(report["rows"]) == len(conditions) * len(methods)


def test_attack_decomposition_rejects_missing_condition() -> None:
    with pytest.raises(MechanismError, match="missing required attack conditions"):
        decompose_attack_ledger([], "a" * 64)


def test_feature_group_mapping_covers_all_112_full_features_once() -> None:
    groups = full_mstf_feature_groups()

    indices = [index for group in groups.values() for index in group]
    assert sorted(indices) == list(range(112))
    assert jaccard_top_k(["a", "b"], ["b", "c"], 10) == 1 / 3


def test_importance_aggregation_normalizes_features_and_groups() -> None:
    names = mstf_feature_names()
    groups = full_mstf_feature_groups()
    gain = [[float(index + 1) for index in range(112)], [float(index + 1) for index in range(112)]]
    permutation = [[0.0] * 111 + [1.0], [0.0] * 111 + [1.0]]

    report = aggregate_importance(gain, permutation, names, groups)

    assert len(report["feature_rows"]) == 112
    assert len(report["group_rows"]) == 16
    assert report["feature_rows"][-1]["feature"] == names[-1]
    assert report["feature_rows"][-1]["gain_mean"] > report["feature_rows"][0]["gain_mean"]
    assert report["feature_rows"][-1]["permutation_mean"] == 1.0
