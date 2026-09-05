"""Contracts for registry and record-count integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lpcode_v1.integrity_audit import IntegrityError, audit_registry_hashes


def test_registry_hash_audit_rejects_mutated_registered_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"bundles": {"gate_a": {"files": {"config.json": {"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "bytes": artifact.stat().st_size}}}}}), encoding="utf-8")
    artifact.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(IntegrityError, match="hash mismatch"):
        audit_registry_hashes(registry)
