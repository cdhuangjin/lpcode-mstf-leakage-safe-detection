"""Contracts for immutable discovery of the four formal Gate bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lpcode_v1.canonical import CanonicalArtifactError, discover_canonical_bundles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path, gate_file: str, protocol: str) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"split_protocol": protocol}), encoding="utf-8")
    (root / "folds.jsonl").write_text('{"fold": 0}\n', encoding="utf-8")
    (root / "summary.json").write_text("{}\n", encoding="utf-8")
    (root / gate_file).write_text(
        json.dumps({"status": "evaluable", "strict": {"passed": True}}), encoding="utf-8"
    )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in root.iterdir()
        if path.name != "manifest.json"
    }
    (root / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def test_discovery_rejects_smoke_and_requires_gate_a_bundle(tmp_path: Path) -> None:
    (tmp_path / "01_transition_test_strict_origins_task2_smoke").mkdir()

    with pytest.raises(CanonicalArtifactError, match="gate_a"):
        discover_canonical_bundles(tmp_path, cross_language_root=tmp_path / "missing")


def test_discovery_records_external_gate_d_root_and_file_hashes(tmp_path: Path) -> None:
    _write_bundle(tmp_path / "01_transition_test_strict_origins", "gate_a.json", "strict-v2")
    _write_bundle(tmp_path / "02_unseen_llm", "gate_b.json", "unseen-v1")
    _write_bundle(tmp_path / "03_style_attack", "gate_c.json", "attack-v1")
    gate_d_root = tmp_path / "external" / "04_cross_language"
    _write_bundle(gate_d_root, "gate_d.json", "cross-language-v1")

    bundles = discover_canonical_bundles(tmp_path, cross_language_root=gate_d_root)

    assert bundles["gate_d"].root == gate_d_root.resolve()
    assert bundles["gate_d"].files["manifest.json"].sha256 == _sha256(gate_d_root / "manifest.json")
    assert bundles["gate_a"].protocol_version == "strict-v2"


def test_discovery_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    roots = (
        ("01_transition_test_strict_origins", "gate_a.json"),
        ("02_unseen_llm", "gate_b.json"),
        ("03_style_attack", "gate_c.json"),
    )
    for name, gate_file in roots:
        _write_bundle(tmp_path / name, gate_file, "protocol")
    gate_d_root = tmp_path / "04_cross_language"
    _write_bundle(gate_d_root, "gate_d.json", "protocol")
    (tmp_path / "02_unseen_llm" / "summary.json").write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(CanonicalArtifactError, match="hash mismatch"):
        discover_canonical_bundles(tmp_path, cross_language_root=gate_d_root)
