"""Contracts for the registry-bound A0--A5 orthogonal ablation."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from lpcode_v1.ablation import (
    ABLATION_METHODS,
    AblationContractError,
    _assert_pair_audit_passes,
    _assert_resume_record,
    _audit_pair_split,
    _build_smoke_config,
    _replace_with_retry,
    _source_index,
    _write_immutable_json,
    assert_pair_contract,
    assert_smoke_output_root,
    verify_registry_files,
)


def test_ablation_matrix_is_fixed_and_xgboost_only() -> None:
    assert ABLATION_METHODS == {
        "A0": {"feature_family": "official10", "feature_count": 10, "representation": "concat"},
        "A1": {"feature_family": "official10", "feature_count": 10, "representation": "concat_delta"},
        "A2": {"feature_family": "enhanced28", "feature_count": 28, "representation": "concat"},
        "A3": {"feature_family": "enhanced28", "feature_count": 28, "representation": "delta"},
        "A4": {"feature_family": "enhanced28", "feature_count": 28, "representation": "concat_delta"},
        "A5": {"feature_family": "enhanced28", "feature_count": 28, "representation": "full"},
    }


def test_pair_contract_rejects_a_reconstructed_split_that_changed() -> None:
    source = {
        "language": "c",
        "seed": 42,
        "fold": 0,
        "train_index_sha256": "a" * 64,
        "test_index_sha256": "b" * 64,
    }
    record = {**source, "train_index_sha256": "c" * 64}

    with pytest.raises(AblationContractError, match="pair hash"):
        assert_pair_contract(source, record)


def test_clean_source_index_does_not_require_a_heldout_llm() -> None:
    record = {
        "language": "c",
        "seed": 42,
        "fold": 0,
        "train_index_sha256": "a" * 64,
        "test_index_sha256": "b" * 64,
    }

    assert _source_index([record], unseen=False) == {("c", None, 42, 0): record}


def test_atomic_replace_retries_transient_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_text("new", encoding="utf-8")
    calls = {"count": 0}

    def transient_replace(left, right):
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("temporary lock")
        left.rename(right)

    monkeypatch.setattr("lpcode_v1.ablation.os.replace", transient_replace)
    _replace_with_retry(source, destination, attempts=2, delay_seconds=0)
    assert destination.read_text(encoding="utf-8") == "new"


def test_smoke_output_must_be_explicit_and_outside_protected_roots(tmp_path) -> None:
    gate_root = tmp_path / "gate_a"
    gate_root.mkdir()

    with pytest.raises(AblationContractError, match="smoke"):
        assert_smoke_output_root(tmp_path / "ordinary-output", (gate_root,))
    with pytest.raises(AblationContractError, match="protected"):
        assert_smoke_output_root(gate_root / "smoke", (gate_root,))

    assert_smoke_output_root(tmp_path / "ablation_smoke", (gate_root,)) == (tmp_path / "ablation_smoke").resolve()


def test_existing_config_is_immutable(tmp_path) -> None:
    path = tmp_path / "config.json"
    _write_immutable_json(path, {"seed": 42})
    _write_immutable_json(path, {"seed": 42})

    with pytest.raises(AblationContractError, match="immutable config"):
        _write_immutable_json(path, {"seed": 123})


def _pair(*, label: int, pair_hash: str, human: str, candidate: str, human_code: str, candidate_code: str):
    return SimpleNamespace(
        label=label,
        pair_sha256=pair_hash,
        human_origin_id=human,
        candidate_origin_id=candidate,
        human_code_sha256=human_code,
        candidate_code_sha256=candidate_code,
        human_component_id=f"component-{human}",
        candidate_component_id=(f"component-{candidate}" if label == 0 else f"component-{human}"),
    )


def _clean_split():
    return SimpleNamespace(
        train_pairs=(
            _pair(label=1, pair_hash="1" * 64, human="h1", candidate="c1", human_code="a" * 64, candidate_code="b" * 64),
            _pair(label=0, pair_hash="2" * 64, human="h2", candidate="c2", human_code="c" * 64, candidate_code="d" * 64),
        ),
        test_pairs=(
            _pair(label=1, pair_hash="3" * 64, human="h3", candidate="c3", human_code="e" * 64, candidate_code="f" * 64),
            _pair(label=0, pair_hash="4" * 64, human="h4", candidate="c4", human_code="0" * 64, candidate_code="9" * 64),
        ),
    )


def test_pair_split_audit_reports_zero_leakage_and_balanced_classes() -> None:
    audit = _audit_pair_split(_clean_split())

    assert audit == {
        "endpoint_leakage_count": 0,
        "content_leakage_count": 0,
        "negative_component_violation_count": 0,
        "duplicate_pair_count": 0,
        "train_class_counts": {"0": 1, "1": 1},
        "test_class_counts": {"0": 1, "1": 1},
    }


def test_pair_audit_is_a_hard_gate_before_model_evaluation() -> None:
    audit = _audit_pair_split(_clean_split())
    audit["content_leakage_count"] = 1

    with pytest.raises(AblationContractError, match="content leakage"):
        _assert_pair_audit_passes(audit)


def test_resume_record_is_checked_against_reconstructed_pair_hashes() -> None:
    source = {
        "language": "c",
        "seed": 42,
        "fold": 0,
        "train_index_sha256": "a" * 64,
        "test_index_sha256": "b" * 64,
        "train_pair_sha256": "a" * 64,
        "test_pair_sha256": "b" * 64,
    }
    existing = {
        **source,
        "method": "A0",
        "feature_dimensions": 20,
        "frozen_registry_sha256": "f" * 64,
    }

    _assert_resume_record(source, existing, method="A0", frozen_registry_sha256="f" * 64)
    existing["test_pair_sha256"] = "c" * 64
    with pytest.raises(AblationContractError, match="resume pair hash"):
        _assert_resume_record(source, existing, method="A0", frozen_registry_sha256="f" * 64)


def test_smoke_config_has_separate_clean_and_unseen_expected_counts() -> None:
    config = _build_smoke_config(
        frozen_registry_sha256="f" * 64,
        negative_pair_artifact_sha256={"config.json": "e" * 64},
        language="c",
        seed=42,
        n_splits=2,
        limit_origins=8,
        heldout_llm="gpt3.5",
    )

    assert config["run_kind"] == "smoke"
    assert config["expected_clean"] == 12
    assert config["expected_unseen"] == 12
    assert config["expected_total"] == 24


def test_registry_file_verification_detects_post_freeze_mutation(tmp_path) -> None:
    controlled = tmp_path / "controlled.json"
    controlled.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(controlled.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"bundles": {"gate_a": {"files": {"controlled.json": {"path": str(controlled), "sha256": digest, "bytes": controlled.stat().st_size}}}}}),
        encoding="utf-8",
    )

    verify_registry_files(registry_path)
    controlled.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(AblationContractError, match="ARTIFACT MUTATION"):
        verify_registry_files(registry_path)
