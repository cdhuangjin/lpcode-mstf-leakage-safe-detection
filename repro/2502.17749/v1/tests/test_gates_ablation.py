"""Protocol gates for A0--A5 ablation ledgers."""

from __future__ import annotations

import copy

import pytest

from lpcode_v1.gates_ablation import (
    AblationAuditError,
    expected_feature_dimension,
    validate_ablation_records,
)


REGISTRY_SHA = "f" * 64
METHOD_DIMENSIONS = {
    "A0": 20,
    "A1": 30,
    "A2": 56,
    "A3": 28,
    "A4": 84,
    "A5": 112,
}


def _record(method: str) -> dict:
    return {
        "environment": "clean",
        "method": method,
        "language": "c",
        "heldout_llm": None,
        "seed": 42,
        "fold": 0,
        "train_pair_sha256": "a" * 64,
        "test_pair_sha256": "b" * 64,
        "train_index_sha256": "a" * 64,
        "test_index_sha256": "b" * 64,
        "feature_dimensions": METHOD_DIMENSIONS[method],
        "frozen_registry_sha256": REGISTRY_SHA,
        "f1": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "auroc": 0.95,
        "mcc": 0.8,
        "endpoint_leakage_count": 0,
        "content_leakage_count": 0,
        "negative_component_violation_count": 0,
        "duplicate_pair_count": 0,
        "train_class_counts": {"0": 8, "1": 8},
        "test_class_counts": {"0": 4, "1": 4},
    }


def _complete_cell() -> list[dict]:
    return [_record(method) for method in METHOD_DIMENSIONS]


def test_method_dimensions_are_derived_from_the_exact_contract() -> None:
    assert {method: expected_feature_dimension(method) for method in METHOD_DIMENSIONS} == METHOD_DIMENSIONS


def test_complete_cell_passes_all_ablation_audits() -> None:
    audit = validate_ablation_records(
        _complete_cell(), expected_count=6, frozen_registry_sha256=REGISTRY_SHA
    )

    assert audit["status"] == "PASS"
    assert audit["record_count"] == 6
    assert audit["cell_count"] == 1


def test_duplicate_fold_record_is_rejected() -> None:
    records = _complete_cell()
    records.append(copy.deepcopy(records[0]))

    with pytest.raises(AblationAuditError, match="duplicate record key"):
        validate_ablation_records(records, expected_count=7, frozen_registry_sha256=REGISTRY_SHA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("f1", float("nan"), "non-finite"),
        ("auroc", float("inf"), "non-finite"),
        ("precision", 1.1, "outside"),
        ("mcc", -1.1, "outside"),
    ],
)
def test_invalid_metrics_fail_the_audit(field: str, value: float, message: str) -> None:
    records = _complete_cell()
    records[0][field] = value

    with pytest.raises(AblationAuditError, match=message):
        validate_ablation_records(records, expected_count=6, frozen_registry_sha256=REGISTRY_SHA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("train_pair_sha256", "c" * 64, "pair hashes"),
        ("test_class_counts", {"0": 5, "1": 3}, "class balance"),
        ("endpoint_leakage_count", 1, "endpoint leakage"),
        ("content_leakage_count", 1, "content leakage"),
        ("negative_component_violation_count", 1, "negative-component"),
        ("duplicate_pair_count", 1, "duplicate pair"),
        ("feature_dimensions", 999, "feature dimension"),
        ("frozen_registry_sha256", "0" * 64, "registry"),
    ],
)
def test_protocol_or_fairness_violation_fails_the_audit(field: str, value, message: str) -> None:
    records = _complete_cell()
    records[-1][field] = value

    with pytest.raises(AblationAuditError, match=message):
        validate_ablation_records(records, expected_count=6, frozen_registry_sha256=REGISTRY_SHA)

