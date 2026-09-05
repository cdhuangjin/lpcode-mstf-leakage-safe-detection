"""Pure validation gates for registry-bound A0--A5 ablation records."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


class AblationAuditError(RuntimeError):
    """Raised when an ablation ledger violates a registered contract."""


METHOD_CONTRACT: dict[str, dict[str, Any]] = {
    "A0": {"feature_family": "official10", "feature_count": 10, "representation": "concat"},
    "A1": {"feature_family": "official10", "feature_count": 10, "representation": "concat_delta"},
    "A2": {"feature_family": "enhanced28", "feature_count": 28, "representation": "concat"},
    "A3": {"feature_family": "enhanced28", "feature_count": 28, "representation": "delta"},
    "A4": {"feature_family": "enhanced28", "feature_count": 28, "representation": "concat_delta"},
    "A5": {"feature_family": "enhanced28", "feature_count": 28, "representation": "full"},
}

_REPRESENTATION_MULTIPLIER = {"concat": 2, "delta": 1, "concat_delta": 3, "full": 4}
_METRIC_RANGES = {
    "f1": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "auroc": (0.0, 1.0),
    "mcc": (-1.0, 1.0),
}
_LEAKAGE_FIELDS = {
    "endpoint_leakage_count": "endpoint leakage",
    "content_leakage_count": "content leakage",
    "negative_component_violation_count": "negative-component violation",
    "duplicate_pair_count": "duplicate pair",
}


def expected_feature_dimension(method: str) -> int:
    """Return the representation dimension implied by the registered method."""
    try:
        spec = METHOD_CONTRACT[method]
    except KeyError as exc:
        raise AblationAuditError(f"unknown ablation method: {method}") from exc
    return int(spec["feature_count"]) * _REPRESENTATION_MULTIPLIER[str(spec["representation"])]


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the unique resume key for one method/fold record."""
    return (
        record.get("environment"),
        record.get("method"),
        record.get("language"),
        record.get("heldout_llm"),
        record.get("seed"),
        record.get("fold"),
    )


def cell_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the cross-method pairing key for one environment/fold cell."""
    return (
        record.get("environment"),
        record.get("language"),
        record.get("heldout_llm"),
        record.get("seed"),
        record.get("fold"),
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise AblationAuditError(f"invalid {label}")
    return value.lower()


def _validated_counts(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"0", "1"}:
        raise AblationAuditError(f"invalid {label} class balance")
    negative, positive = value["0"], value["1"]
    if type(negative) is not int or type(positive) is not int or negative <= 0 or positive <= 0 or negative != positive:
        raise AblationAuditError(f"invalid {label} class balance")
    return negative, positive


def _validate_record(record: dict[str, Any], frozen_registry_sha256: str) -> None:
    method = record.get("method")
    if method not in METHOD_CONTRACT:
        raise AblationAuditError(f"unknown ablation method: {method}")
    if record.get("feature_dimensions") != expected_feature_dimension(str(method)):
        raise AblationAuditError(f"feature dimension mismatch for {method}")
    if _require_sha256(record.get("frozen_registry_sha256"), "registry SHA-256") != frozen_registry_sha256:
        raise AblationAuditError("record belongs to a different frozen registry")
    train_hash = _require_sha256(record.get("train_pair_sha256"), "train pair SHA-256")
    test_hash = _require_sha256(record.get("test_pair_sha256"), "test pair SHA-256")
    if record.get("train_index_sha256") != train_hash or record.get("test_index_sha256") != test_hash:
        raise AblationAuditError("pair hashes disagree with compatibility aliases")
    for metric, (low, high) in _METRIC_RANGES.items():
        try:
            value = float(record[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise AblationAuditError(f"missing or invalid metric: {metric}") from exc
        if not math.isfinite(value):
            raise AblationAuditError(f"non-finite metric: {metric}")
        if not low <= value <= high:
            raise AblationAuditError(f"metric outside valid range: {metric}")
    for field, label in _LEAKAGE_FIELDS.items():
        if record.get(field) != 0:
            raise AblationAuditError(f"non-zero {label}")
    _validated_counts(record.get("train_class_counts"), "train")
    _validated_counts(record.get("test_class_counts"), "test")


def validate_ablation_records(
    records: Iterable[dict[str, Any]], *, expected_count: int, frozen_registry_sha256: str
) -> dict[str, Any]:
    """Validate uniqueness, metrics, fairness, and exact cross-method pairing."""
    expected_registry = _require_sha256(frozen_registry_sha256, "registry SHA-256")
    materialized = list(records)
    if len(materialized) != expected_count:
        raise AblationAuditError(f"record count mismatch: expected {expected_count}, got {len(materialized)}")
    seen: set[tuple[Any, ...]] = set()
    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in materialized:
        if not isinstance(record, dict):
            raise AblationAuditError("ablation record must be an object")
        key = record_key(record)
        if key in seen:
            raise AblationAuditError(f"duplicate record key: {key}")
        seen.add(key)
        _validate_record(record, expected_registry)
        by_cell[cell_key(record)].append(record)
    for key, cell in by_cell.items():
        methods = {record["method"] for record in cell}
        if methods != set(METHOD_CONTRACT):
            raise AblationAuditError(f"incomplete method cell: {key}")
        train_hashes = {record["train_pair_sha256"] for record in cell}
        test_hashes = {record["test_pair_sha256"] for record in cell}
        if len(train_hashes) != 1 or len(test_hashes) != 1:
            raise AblationAuditError(f"cross-method pair hashes differ: {key}")
        train_counts = {tuple(sorted(record["train_class_counts"].items())) for record in cell}
        test_counts = {tuple(sorted(record["test_class_counts"].items())) for record in cell}
        if len(train_counts) != 1 or len(test_counts) != 1:
            raise AblationAuditError(f"cross-method class balance differs: {key}")
    return {
        "status": "PASS",
        "record_count": len(materialized),
        "cell_count": len(by_cell),
        "method_count": len(METHOD_CONTRACT),
        "frozen_registry_sha256": expected_registry,
        "feature_dimensions": {method: expected_feature_dimension(method) for method in METHOD_CONTRACT},
        "pair_hash_equality": True,
        "class_balance_equal": True,
        "endpoint_leakage_count": 0,
        "content_leakage_count": 0,
        "negative_component_violation_count": 0,
        "duplicate_pair_count": 0,
    }

