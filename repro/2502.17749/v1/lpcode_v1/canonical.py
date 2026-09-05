"""Discover and freeze the immutable, formal LPcode/MSTF Gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import EXPECTED_COMMIT, build_official_manifest, sha256_file
from .paths import RESULTS_ROOT, WORKSPACE_ROOT


class CanonicalArtifactError(RuntimeError):
    """Raised when a purported formal Gate bundle cannot be frozen safely."""


@dataclass(frozen=True)
class FrozenFile:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class CanonicalBundle:
    name: str
    root: Path
    protocol_version: str
    files: dict[str, FrozenFile]


_BUNDLES = {
    "gate_a": ("01_transition_test_strict_origins", "gate_a.json"),
    "gate_b": ("02_unseen_llm", "gate_b.json"),
    "gate_c": ("03_style_attack", "gate_c.json"),
}
_FORBIDDEN_TOKENS = ("smoke", "preflight", "legacy", "interrupted", "retry")
_REQUIRED_COMMON = ("config.json", "folds.jsonl", "summary.json", "manifest.json")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalArtifactError(f"{label}: unreadable JSON at {path}") from exc
    if not isinstance(value, dict):
        raise CanonicalArtifactError(f"{label}: JSON root must be an object at {path}")
    return value


def _ensure_safe_root(name: str, root: Path) -> None:
    if any(token in part.lower() for part in root.parts for token in _FORBIDDEN_TOKENS):
        raise CanonicalArtifactError(f"{name}: forbidden non-formal artifact path: {root}")


def _protocol(config: dict[str, Any], gate: dict[str, Any], name: str) -> str:
    value = config.get("split_protocol") or config.get("protocol_version") or gate.get("protocol_version")
    if not isinstance(value, str) or not value:
        raise CanonicalArtifactError(f"{name}: missing protocol version")
    return value


def _validate_bundle(name: str, root: Path, gate_filename: str) -> CanonicalBundle:
    root = root.resolve()
    _ensure_safe_root(name, root)
    required = (*_REQUIRED_COMMON, gate_filename)
    for filename in required:
        if not (root / filename).is_file():
            raise CanonicalArtifactError(f"{name}: missing required {filename} in {root}")

    manifest = _load_json(root / "manifest.json", name)
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise CanonicalArtifactError(f"{name}: manifest has no files map")
    for filename in required:
        if filename == "manifest.json":
            continue
        expected = manifest_files.get(filename)
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise CanonicalArtifactError(f"{name}: manifest lacks hash for {filename}")
        actual_hash = sha256_file(root / filename)
        if actual_hash != expected["sha256"]:
            raise CanonicalArtifactError(f"{name}: hash mismatch for {filename}")
        if expected.get("bytes") != (root / filename).stat().st_size:
            raise CanonicalArtifactError(f"{name}: byte-count mismatch for {filename}")

    gate = _load_json(root / gate_filename, name)
    if gate.get("status") != "evaluable" or gate.get("strict", {}).get("passed") is not True:
        raise CanonicalArtifactError(f"{name}: gate is not strict PASS")
    config = _load_json(root / "config.json", name)
    files = {
        filename: FrozenFile(
            path=(root / filename).as_posix(),
            sha256=sha256_file(root / filename),
            bytes=(root / filename).stat().st_size,
        )
        for filename in required
    }
    return CanonicalBundle(name=name, root=root, protocol_version=_protocol(config, gate, name), files=files)


def discover_canonical_bundles(
    results_root: Path, *, cross_language_root: Path
) -> dict[str, CanonicalBundle]:
    """Return four validated formal bundles; never discover smoke/preflight outputs."""
    results_root = results_root.resolve()
    bundles = {
        name: _validate_bundle(name, results_root / dirname, gate_filename)
        for name, (dirname, gate_filename) in _BUNDLES.items()
    }
    bundles["gate_d"] = _validate_bundle("gate_d", cross_language_root, "gate_d.json")
    return bundles


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "scikit-learn", "tree-sitter", "tree-sitter-c", "tree-sitter-cpp", "tree-sitter-java", "xgboost")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def write_frozen_registry(
    output: Path, bundles: dict[str, CanonicalBundle], *, generated_at: datetime | None = None
) -> Path:
    """Write a registry that binds all later work to the verified Gate inputs."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "workspace_root": WORKSPACE_ROOT.as_posix(),
        "official_commit_expected": EXPECTED_COMMIT,
        "official_baseline": build_official_manifest(),
        "bundles": {
            name: {
                "root": bundle.root.as_posix(),
                "protocol_version": bundle.protocol_version,
                "files": {filename: asdict(file) for filename, file in bundle.files.items()},
            }
            for name, bundle in bundles.items()
        },
        "environment": {"python": platform.python_version(), "packages": _package_versions()},
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    return output


def write_current_state(output: Path, bundles: dict[str, CanonicalBundle], registry: Path) -> Path:
    """Write a concise state report without modifying any Gate bundle."""
    rows = ["# LPcode/MSTF Current State", "", "## Frozen formal gates", ""]
    for name, bundle in bundles.items():
        rows.append(f"- **{name.upper()}**: PASS; protocol `{bundle.protocol_version}`; root `{bundle.root}`.")
    rows.extend(("", f"Frozen registry: `{registry}`.", "", "No Gate A-D files were modified."))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--cross-language-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=RESULTS_ROOT / "06_paper_assets" / "frozen_result_registry.json")
    parser.add_argument("--state-output", type=Path, default=RESULTS_ROOT / "08_submission_audit" / "CURRENT_STATE.md")
    parser.add_argument("--write-state", action="store_true")
    args = parser.parse_args()
    bundles = discover_canonical_bundles(args.results_root, cross_language_root=args.cross_language_root)
    registry = write_frozen_registry(args.registry, bundles)
    if args.write_state:
        write_current_state(args.state_output, bundles, registry)
    print(registry)


if __name__ == "__main__":
    main()
